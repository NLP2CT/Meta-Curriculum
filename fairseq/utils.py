# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import defaultdict, namedtuple
from typing import Callable, List
import copy
import importlib.util
import os
import sys
import warnings

import ipdb
from numpy.core.fromnumeric import argsort
import torch
import torch.nn.functional as F

from fairseq.modules import gelu, gelu_fast

BleuStatistics = namedtuple('BleuStatistics', 'correct total sys_len ref_len')
Task = namedtuple('Task', 'train dev test')
Cluster = namedtuple('Cluster', 'id task')
Pair = namedtuple('Pair', 'source target')


def reduce_average_meter(meter_a, meter_b):
    from fairseq.meters import AverageMeter
    return_meter = AverageMeter()
    return_meter.count = meter_a.count + meter_b.count
    return_meter.sum = meter_a.sum + meter_b.sum
    return_meter.sum_square = meter_a.sum_square + meter_b.sum_square
    return return_meter


def reduce_bleu_stats(stats_a: BleuStatistics, stats_b: BleuStatistics):
    return BleuStatistics(correct=reduce_lists(stats_a.correct, stats_b.correct),
                          total=reduce_lists(stats_a.total, stats_b.total),
                          sys_len=stats_a.sys_len+stats_b.sys_len,
                          ref_len=stats_a.ref_len+stats_b.ref_len)


def reduce_lists(list_a, list_b):
    assert len(list_a) == len(list_b)
    return [val_a + val_b for val_a, val_b in zip(list_a, list_b)]


def get_zero_bleu_stats():
    correct = [0.0] * 4
    total = [0.0] * 4
    sys_len = 0.0
    ref_len = 0.0
    return BleuStatistics(correct=correct, total=total, sys_len=sys_len, ref_len=ref_len)


def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
    from fairseq import checkpoint_utils
    deprecation_warning(
        'utils.load_ensemble_for_inference is deprecated. '
        'Please use checkpoint_utils.load_model_ensemble instead.'
    )
    return checkpoint_utils.load_model_ensemble(
        filenames, arg_overrides=model_arg_overrides, task=task,
    )


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str):
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, 'r') as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    print("| Found {}/{} types in embedding file.".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer
    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenizer.tokenize_line(src_str) + ['<eos>']
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return ' '.join(hypo_tokens)


def post_process_prediction_fixed_dict(hypo_tokens, src_str, alignment, align_dict, tgt_dict, remove_bpe):
    # from fairseq import tokenizer
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=False)
    return hypo_tokens, hypo_str, alignment


def post_process_prediction(hypo_tokens, src_str, alignment, align_dict, tgt_dict, remove_bpe):
    # from fairseq import tokenizer
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def make_positions(tensor, padding_idx, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).long()
    return torch.cumsum(mask, dim=1) * mask + padding_idx


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(src_tokens, padding_idx, right_to_left=False, left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def clip_grad_norm_(tensor, max_norm):
    grad_norm = item(torch.norm(tensor))
    if grad_norm > max_norm > 0:
        clip_coef = max_norm / (grad_norm + 1e-6)
        tensor.mul_(clip_coef)
    return grad_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(
                    map(nullsafe_min, zip(max_positions, arg))
                )

    return max_positions


def import_user_module(args):
    module_path = getattr(args, 'user_dir', None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return gelu
    elif activation == 'gelu_fast':
        return gelu_fast
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")


def construct_path(base_dir, split, language):
    return os.path.join(base_dir, split + '.' + language)


def get_source_target_and_user_files(data_dir, subset, source_lang, target_lang):
    source_file = construct_path(base_dir=data_dir, split=subset, language=source_lang)
    target_file = construct_path(base_dir=data_dir, split=subset, language=target_lang)
    user_file = construct_path(base_dir=data_dir, split=subset, language='usr')
    return source_file, target_file, user_file


def get_data_frame(source_file, target_file, user_file, split):
    import pandas as pd
    with open(source_file, 'r', encoding='utf-8') as src_reader:
        with open(target_file, 'r', encoding='utf-8') as target_reader:
            with open(user_file, 'r', encoding='utf-8') as user_reader:
                src_lines = src_reader.readlines()
                target_lines = target_reader.readlines()
                user_lines = user_reader.readlines()
                # Create a dataframe from the three list
                data_frame = pd.DataFrame({'user': user_lines, 'source': src_lines, 'target': target_lines,
                                           'split': split})
                return data_frame.reset_index(drop=True)


def split_and_directory_to_sentences(data_dir, split, source_lang, target_lang):
    source_file, target_file, user_file = get_source_target_and_user_files(
        data_dir=data_dir, subset=split, source_lang=source_lang, target_lang=target_lang)
    return get_data_frame(source_file=source_file, target_file=target_file, user_file=user_file, split=split)


def split_train_dev_test(array, train_percentage, dev_percentage):
    num_of_meta_examples = len(array)
    meta_train_size = int(train_percentage * num_of_meta_examples)
    meta_dev_test_size = num_of_meta_examples - meta_train_size
    meta_dev_size = int(dev_percentage * meta_dev_test_size)
    meta_test_size = meta_dev_test_size - meta_dev_size
    assert meta_train_size + meta_dev_size + meta_test_size == num_of_meta_examples
    # Now create meta-train, meta-dev, and meta-test splits
    meta_train = array[:meta_train_size]
    meta_dev = array[meta_train_size:meta_train_size + meta_dev_size]
    meta_test = array[meta_train_size + meta_dev_size:]
    assert len(meta_train) == meta_train_size
    assert len(meta_dev) == meta_dev_size
    assert len(meta_test) == meta_test_size
    return meta_train, meta_dev, meta_test


# TODO Can also use sklearn to create these splits!
# TODO What is the best split?
# TODO remove rng
def split_meta_tasks(train_sentences, dev_sentences, test_sentences, rng, train_percentage=0.8, dev_percentage=1.0):
    import pandas as pd
    all_sentences = pd.concat([train_sentences, dev_sentences, test_sentences])
    all_tasks = list(all_sentences.groupby('user'))
    all_tasks = [task[1].reset_index(drop=True) for task in all_tasks]
    meta_train, meta_dev, meta_test = split_train_dev_test(array=all_tasks, train_percentage=train_percentage,
                                                           dev_percentage=dev_percentage)
    return meta_train, meta_dev, meta_test


def split_and_directory_to_meta_tasks(data_dir, train_subset, dev_subset, test_subset, source_lang, target_lang, rng,
                                      train_percentage=0.8, dev_percentage=1.0):
    train_sentences = split_and_directory_to_sentences(data_dir=data_dir, split=train_subset, source_lang=source_lang,
                                                       target_lang=target_lang)
    dev_sentences = split_and_directory_to_sentences(data_dir=data_dir, split=dev_subset, source_lang=source_lang,
                                                     target_lang=target_lang)
    test_sentences = split_and_directory_to_sentences(data_dir=data_dir, split=test_subset, source_lang=source_lang,
                                                      target_lang=target_lang)
    return split_meta_tasks(train_sentences=train_sentences, dev_sentences=dev_sentences, test_sentences=test_sentences,
                            rng=rng, train_percentage=train_percentage, dev_percentage=dev_percentage)

# Converts user translation task data to fairseq task
def to_fairseq_task(task, args, src_dict, tgt_dict):
    from fairseq import tasks
    from copy import deepcopy
    # task is a Tuple, first entry is the key
    task_args = deepcopy(args)
    task_args.train_subset = task_args.train_subset + '.' + task.user.any().strip()
    task_args.valid_subset = task_args.valid_subset + '.' + task.user.any().strip()
    task_args.test_subset = task_args.test_subset + '.' + task.user.any().strip()
    task.split = task.split + '.' + task.user.any().strip()
    fairseq_task = tasks.setup_task(args=task_args, user_data_frame=task, src_dict=src_dict, tgt_dict=tgt_dict)
    return fairseq_task


def run_inference_on_sample(sample, use_cuda, args, gen_timer, task, generator, model, tgt_dict, align_dict, subset,
                            src_dict, scorer, wps_meter):
    assert 'net_input' in sample

    score_sum = 0.
    count = 0
    num_sentences = 0

    sample = move_to_cuda(sample) if use_cuda else sample

    prefix_tokens = None
    if args.prefix_size > 0:
        prefix_tokens = sample['target'][:, :args.prefix_size]

    gen_timer.start()
    hypos = task.inference_step(generator, [model], sample, prefix_tokens)
    num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
    gen_timer.stop(num_generated_tokens)

    for i, sample_id in enumerate(sample['id'].tolist()):
        has_target = sample['target'] is not None and args.task != 'sine'

        # Remove padding
        src_tokens = strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
        target_tokens = None
        if has_target:
            target_tokens = strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

        # Either retrieve the original sentences or regenerate them from tokens.
        if align_dict is not None:
            src_str = task.dataset(subset).src.get_original_text(sample_id)
            target_str = task.dataset(subset).tgt.get_original_text(sample_id)
        else:
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
            else:
                src_str = ""
            if has_target:
                target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

        if not args.quiet:
            if src_dict is not None:
                print('S-{}\t{}'.format(sample_id, src_str))
            if has_target:
                print('T-{}\t{}'.format(sample_id, target_str))

        # Process top predictions
        for j, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):

            pos_scores = hypo['positional_scores'].float()
            score_sum += pos_scores.sum().cpu()
            count += pos_scores.numel()

            hypo_tokens, hypo_str, alignment = post_process_prediction_fixed_dict(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )

            if not args.quiet:
                print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                print('P-{}\t{}'.format(
                    sample_id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        hypo['positional_scores'].tolist(),
                    ))
                ))

                if args.print_alignment:
                    print('A-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(lambda x: str(item(x)), alignment))
                    ))

            # Score only the top hypothesis
            if has_target and j == 0:
                if align_dict is not None or args.remove_bpe is not None:
                    # Convert back to tokens for evaluation with unk replacement and/or without BPE
                    target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=False)
                if hasattr(scorer, 'add_string'):
                    scorer.add_string(target_str, hypo_str)
                else:
                    scorer.add(target_tokens, hypo_tokens)

    wps_meter.update(num_generated_tokens)
    num_sentences += sample['nsentences']
    return score_sum, count, num_sentences

# TODO minimize dependencies as much as we can!
def run_inference(args, task, model, subset):
    from fairseq.meters import StopwatchMeter, TimeMeter
    from fairseq import bleu
    from fairseq import progress_bar
    import numpy as np
    import math
    """infers from the model on the validation set(s) and return the predictions."""

    score_sum = 0.
    count = 0

    assert not args.sampling or args.nbest == args.beam, '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, '--replace-unk requires a raw text dataset (--raw-text)'
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        model.cuda()
    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    task.load_dataset(subset)
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=resolve_max_positions(
            task.max_positions(),
            model.max_positions()
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        if hasattr(args, 'lowercase'):
            scorer = bleu.SacrebleuScorer(lowercase=args.lowercase)
        else:
            scorer = bleu.SacrebleuScorer()
    else:
        # scorer = bleu.SacrebleuScorer()
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            score_sum_per_sample, count_per_sample, num_sentences_per_sample = run_inference_on_sample(
                sample, use_cuda, args, gen_timer, task, generator, model, tgt_dict, align_dict, subset, src_dict,
                scorer, wps_meter)
            score_sum += score_sum_per_sample
            count += count_per_sample
            num_sentences += num_sentences_per_sample

    if not args.quiet:
        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
        avg_nll_loss = -score_sum / count
        print('| Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
        print('| Loss: {:.4f}, Perplexity: {:.2f}'.format(avg_nll_loss/math.log(2), np.exp(avg_nll_loss)))
    result = None
    bleu_stats = get_zero_bleu_stats()
    if has_target:
        result_string = scorer.result_string()
        bleu_stats = BleuStatistics(correct=result_string.counts, total=result_string.totals,
                                    sys_len=result_string.sys_len, ref_len=result_string.ref_len)
        if not args.quiet:
            print('| Generate {} with beam={}: {}'.format(subset, args.beam, result_string))
        result = result_string.score
    return result, bleu_stats


def create_epoch_iterator(task, dataset, args, max_positions):
    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    )
    return epoch_itr


# TODO copied, see below
def get_perplexity(loss):
    import math
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


# TODO copied, see below
def save_checkpoint(args, trainer, epoch_itr, val_loss):
    from fairseq import distributed_utils
    import collections
    from fairseq.meters import StopwatchMeter
    from fairseq import checkpoint_utils
    if args.no_save or not distributed_utils.is_master(args):
        return

    write_timer = StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

        write_timer.stop()
        print('| saved checkpoint {} (epoch {} @ {} updates) (writing took {} seconds)'.format(
            checkpoints[0], epoch, updates, write_timer.sum))

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_utils.checkpoint_paths(
            args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_utils.checkpoint_paths(
            args.save_dir, pattern=r'checkpoint(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


# TODO also copied from train.py, see below
def get_valid_stats(trainer, tag='valid'):
    import collections
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter(tag + '_loss')
    if trainer.get_meter(tag + '_nll_loss').count > 0:
        nll_loss = trainer.get_meter(tag + '_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best_loss'] = min(save_checkpoint.best, stats['loss'].avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    return stats


# TODO this is copied from train.py, didn't want to change fairseq codebase, but we can just import and use this
# function in train.py too!
def validate(args, trainer, task, epoch_itr, subsets, train_progress=None):
    import collections
    from fairseq import progress_bar
    from fairseq.meters import AverageMeter, BleuMeter
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    valid_stats = []
    for subset in subsets:
        # Initialize data iterator
        # ipdb.set_trace()
        task.load_dataset(subset)
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())
        # TODO Handle lowercase
        # TODO abstract away this part of the code that's common with train, validate, etc...
        extra_meters['bleu'] = BleuMeter()

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
            stats[k+'_std'] = meter.std
        if train_progress is not None:
            train_progress.print(stats, tag=subset, step=trainer.get_num_updates())
            train_progress.flush(tag=subset)
        else:
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            progress.flush(tag=subset)
        valid_stats.append({'num_updates': stats['num_updates'], 'loss': stats['loss'].avg, 'bleu': extra_meters['bleu']})
        valid_losses.append(stats['loss'].avg)
    return valid_losses, valid_stats

def forward_and_accumulate_gradient(args, trainer, task, epoch_itr, subset, train_progress=None):
    import math
    from fairseq import progress_bar
    trainer.optimizer.zero_grad()
    task.load_dataset(subset)
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple'
    )
    task_loss = 0.0
    count = 0.0
    for sample in progress:
        loss, sample_size, log_output = trainer.forward_step(sample)
        forward_loss = (loss / math.log(2)) / float(sample_size)
        forward_loss.backward()
        task_loss += log_output['loss']
        count += 1
    if count > 0:
        task_loss = task_loss / float(count)
    return task_loss


# TODO Abstract what is common with validate()
def forward(args, trainer, task, epoch_itr, subset, train_progress=None):
    import collections
    import math
    from fairseq import progress_bar
    from fairseq.meters import AverageMeter, BleuMeter
    """Evaluate the model on the validation set(s) and return the losses."""
    # Initialize data iterator
    task.load_dataset(subset)
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple'
    )
    # reset forward loss meters
    for k in ['forward_loss', 'forward_nll_loss']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    # TODO handle lowercase
    # TODO abstract away what's common with train, validate, etc...
    extra_meters['bleu'] = BleuMeter()

    forward_loss_sum = torch.zeros(1, requires_grad=True)
    if trainer.cuda:
        forward_loss_sum = forward_loss_sum.cuda()
    forward_loss_count = 0.0
    for sample in progress:
        loss, sample_size, log_output = trainer.forward_step(sample)
        forward_loss_sum = forward_loss_sum + (loss / math.log(2))
        forward_loss_count = forward_loss_count + sample_size
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue
            extra_meters[k].update(v)
    # log validation stats
    stats = get_valid_stats(trainer, tag='forward')
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
        stats[k+'_std'] = meter.std
    if train_progress is not None:
        train_progress.print(stats, tag=subset, step=trainer.get_num_updates())
        train_progress.flush(tag=subset)
    else:
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        progress.flush(tag=subset)
    if forward_loss_count > 0:
        forward_loss = forward_loss_sum / float(forward_loss_count)
    else:
        forward_loss = forward_loss_sum
    return forward_loss

# TODO this is also copied
def train_buckets(args, trainer, task, epoch_itr, is_curriculum):
    import collections
    import math
    from fairseq.data import iterators
    from fairseq import progress_bar
    from fairseq.meters import AverageMeter, BleuMeter
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    assert (args.is_curriculum == is_curriculum)
    # Initialize data iterator
    if args.is_curriculum:
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=False,
        )
    else:
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.epoch >= args.curriculum),
        )
    # print('is curri?', shuffle_close)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )
    # TODO:  for debuging, Remove
    if task.args.task == 'user_translation':
        print('{} is curriculum? {}'.format(task.user_data_frame.task_group.any().strip(), task.args.is_curriculum))
        print('Downstream score for debug: {}'.format(task.user_data_frame[:3].score))
    if task.args.task == 'meta-curriculum':
        print('Meta-Train is curriculum? {}'.format(task.is_curriculum))
        print('Meta-Train score for debug: {}'.format([i.task_score for i in task.meta_train_tasks]))
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    # TODO handle lowercase
    # TODO abstract away these two lines from train, validate, ... etc
    extra_meters['bleu'] = BleuMeter()

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    batch_info = []
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
            stats[k+'_std'] = extra_meters[k].std
        progress.log(stats, tag=args.train_subset, step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets, train_progress=progress)
            batch_info.append(valid_stats)
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
        stats[k+'_std'] = meter.std
    progress.print(stats, tag=args.train_subset, step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()
    return batch_info

def bucket_sgd(task, args, model, criterion, bucket, train_function, cur_epoch, last_trainer=None):
    import math
    from fairseq.trainer import Trainer
    train_dataset = task.datasets[args.train_subset+'_bucket'][bucket]
    print("| train bucket {} on {} sents".format(bucket, len(train_dataset)))
    # Make a dummy batch to (i) warm the caching allocator and (ii) as a  placeholder DistributedDataParallel when
    # there's an uneven number of batches per worker.
    max_positions = resolve_max_positions(task.max_positions(), model.max_positions(),)
    dummy_batch = train_dataset.get_dummy_batch(num_tokens=args.max_tokens, max_positions=max_positions)
    oom_batch = task.dataset(args.train_subset).get_dummy_batch(1, max_positions)
    # Create a trainer for training the model
    if last_trainer is None:
        trainer = Trainer(args, task, model, criterion, dummy_batch, oom_batch)
    else:
        trainer = last_trainer
    epoch_itr = create_epoch_iterator(task=task, dataset=train_dataset, args=args, max_positions=max_positions)
    epoch_itr.epoch = cur_epoch
    print('| Finetune epoch: ', epoch_itr.epoch)
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    # Do SGD on this task
    valid_subsets = args.valid_subset.split(',')
    lr = trainer.get_lr()
    # TODO batch_info could be misleading if number of epochs > 1
    batch_info = []
    # Always validate once before training
    valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)
    batch_info.append(valid_stats)
    inner_epoch = 0
    max_inner_epoch = 1
    while lr > args.min_lr and inner_epoch < max_inner_epoch and trainer.get_num_updates() < max_update:
        # Train the model for one epoch
        inner_epoch += 1
        batch_info += train_function(args=args, trainer=trainer, task=task, epoch_itr=epoch_itr, is_curriculum=args.is_curriculum)
        # Evaluate on validation split
        # if epoch_itr.epoch % args.validate_interval == 0:
        #     valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)
        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
    if batch_info is None:
        # Handle the original train function
        batch_info = []
    sgd_dict = {'trainer': trainer, 'epoch_itr': epoch_itr, 'batch_info': batch_info, 'valid_losses': valid_losses,
                'valid_stats': valid_stats}
    return sgd_dict


def sgd(task, args, model, criterion, train_function):
    import math
    from fairseq.trainer import Trainer
    task.load_dataset(args.train_subset)
    train_dataset = task.dataset(args.train_subset)
    print("| train on {} sents".format(len(train_dataset)))
    # Make a dummy batch to (i) warm the caching allocator and (ii) as a  placeholder DistributedDataParallel when
    # there's an uneven number of batches per worker.
    max_positions = resolve_max_positions(task.max_positions(), model.max_positions(),)
    dummy_batch = train_dataset.get_dummy_batch(num_tokens=args.max_tokens, max_positions=max_positions)
    oom_batch = task.dataset(args.train_subset).get_dummy_batch(1, max_positions)
    # Create a trainer for training the model
    trainer = Trainer(args, task, model, criterion, dummy_batch, oom_batch)
    epoch_itr = create_epoch_iterator(task=task, dataset=train_dataset, args=args, max_positions=max_positions)
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    # Do SGD on this task
    valid_subsets = args.valid_subset.split(',')
    lr = trainer.get_lr()
    # TODO batch_info could be misleading if number of epochs > 1
    batch_info = []
    # Always validate once before training
    valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)
    batch_info.append(valid_stats)
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # Train the model for one epoch
        batch_info += train_function(args=args, trainer=trainer, task=task, epoch_itr=epoch_itr, is_curriculum=args.is_curriculum)
        # Evaluate on validation split
        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)
        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
    if batch_info is None:
        # Handle the original train function
        batch_info = []
    sgd_dict = {'trainer': trainer, 'epoch_itr': epoch_itr, 'batch_info': batch_info, 'valid_losses': valid_losses,
                'valid_stats': valid_stats}
    return sgd_dict


# TODO This is copied
def get_training_stats(trainer):
    import collections
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


# TODO this is also copied
def train(args, trainer, task, epoch_itr, is_curriculum):
    import collections
    import math
    from fairseq.data import iterators
    from fairseq import progress_bar
    from fairseq.meters import AverageMeter, BleuMeter
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    assert (args.is_curriculum == is_curriculum)
    # Initialize data iterator
    if args.is_curriculum:
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=False,
        )
    else:
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.epoch >= args.curriculum),
        )
    # print('is curri?', shuffle_close)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )
    # TODO:  for debuging, Remove
    if task.args.task == 'user_translation':
        print('{} is curriculum? {}'.format(task.user_data_frame.task_group.any().strip(), task.args.is_curriculum))
        print('Downstream score for debug: {}'.format(task.user_data_frame[:3].score))
    if task.args.task == 'meta-curriculum':
        print('Meta-Train is curriculum? {}'.format(task.is_curriculum))
        print('Meta-Train score for debug: {}'.format([i.task_score for i in task.meta_train_tasks]))
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    # TODO handle lowercase
    # TODO abstract away these two lines from train, validate, ... etc
    extra_meters['bleu'] = BleuMeter()

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    batch_info = []
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
            stats[k+'_std'] = extra_meters[k].std
        progress.log(stats, tag=args.train_subset, step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets, train_progress=progress)
            batch_info.append(valid_stats)
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
        stats[k+'_std'] = meter.std
    progress.print(stats, tag=args.train_subset, step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()
    return batch_info

def fine_tune_on_task(model, task, args):
    from copy import deepcopy
    # save snapshot before evaluation
    weights_before = deepcopy(model.state_dict())
    # train on meta-test task
    criterion = task.build_criterion(args)
    sgd_dict = sgd(task=task, args=args, model=model, criterion=criterion, train_function=train)
    loss_val, _ = validate(args=args, trainer=sgd_dict['trainer'], task=task, epoch_itr=sgd_dict['epoch_itr'],
                           subsets=args.valid_subset.split(','))[0]
    _, bleu = run_inference(args=args, task=task, model=model, subset=args.valid_subset)
    # restore from snapshot
    model.load_state_dict(weights_before)
    return loss_val, bleu, sgd_dict['batch_info']


def get_writers_by_split(split, split_dir, source_lang, target_lang):
    import pathlib
    pathlib.Path(split_dir).mkdir(parents=True, exist_ok=True)
    source_file = os.path.join(split_dir, '{}.{}'.format(split, source_lang))
    target_file = os.path.join(split_dir, '{}.{}'.format(split, target_lang))
    user_file = os.path.join(split_dir, '{}.{}'.format(split, 'usr'))
    english_writer = open(source_file, 'w', encoding='utf-8')
    german_writer = open(target_file, 'w', encoding='utf-8')
    user_writer = open(user_file, 'w', encoding='utf-8')
    return english_writer, german_writer, user_writer


def write_sentences(split: Pair, en_writer, de_writer, user_writer, cluster_id):
    english_sentences, german_sentences = split
    for english_sentence, german_sentence in zip(english_sentences, german_sentences):
        en_writer.write(english_sentence)
        de_writer.write(german_sentence)
        user_writer.write(str(cluster_id) + '\n')


def write_split_data(clusters: Task, split_dir, source_lang, target_lang, train_name='train', dev_name='dev',
                     test_name='test'):
    # We need three writers for train / dev / test
    train_en_writer, train_de_writer, train_user_writer = get_writers_by_split(
        split=train_name, split_dir=split_dir, source_lang=source_lang, target_lang=target_lang)
    dev_en_writer, dev_de_writer, dev_user_writer = get_writers_by_split(
        split=dev_name, split_dir=split_dir, source_lang=source_lang, target_lang=target_lang)
    test_en_writer, test_de_writer, test_user_writer = get_writers_by_split(
        split=test_name, split_dir=split_dir, source_lang=source_lang, target_lang=target_lang)
    train_clusters: List[Cluster] = clusters.train
    test_clusters: List[Cluster] = clusters.test
    dev_clusters: List[Cluster] = clusters.dev
    assert len(train_clusters) == len(test_clusters)
    assert len(test_clusters) == len(dev_clusters)
    for cluster_id, train_sentences in train_clusters:
        write_sentences(train_sentences, train_en_writer, train_de_writer, train_user_writer, cluster_id)
    for cluster_id, test_sentences in test_clusters:
        write_sentences(test_sentences, test_en_writer, test_de_writer, test_user_writer, cluster_id)
    for cluster_id, dev_sentences in dev_clusters:
        write_sentences(dev_sentences, dev_en_writer, dev_de_writer, dev_user_writer, cluster_id)
