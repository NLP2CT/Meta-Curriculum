import math
import os
import random
import sys
import ipdb
import copy

import numpy as np
import pandas as pd
import torch

from fairseq import utils, options, distributed_utils, tasks
from fairseq.checkpoint_utils import load_model_ensemble, load_checkpoint_to_cpu
from fairseq.trainer import Trainer


def maybe_validate(meta_epoch_itr, meta_learning_args, meta_trainer, meta_learning_task, valid_subsets):
    if meta_epoch_itr.epoch % meta_learning_args.validate_interval == 0:
        return utils.validate(meta_learning_args, meta_trainer, meta_learning_task, meta_epoch_itr, valid_subsets)
    else:
        return [0.0], None

def prepare_meta_task(model, meta_learning_task, meta_learning_args, meta_learning_criterion):
    meta_learning_task.load_dataset(split=meta_learning_args.train_subset)
    train_dataset = meta_learning_task.dataset(meta_learning_args.train_subset)
    # Make a dummy batch to (i) warm the caching allocator and (ii) as a  placeholder DistributedDataParallel when
    # there's an uneven number of batches per worker.
    max_positions = utils.resolve_max_positions(meta_learning_task.max_positions(), model.max_positions(),)
    dummy_batch = train_dataset.get_dummy_batch(num_tokens=meta_learning_args.max_tokens, max_positions=max_positions)
    oom_batch = meta_learning_task.dataset(meta_learning_args.train_subset).get_dummy_batch(1, max_positions)
    # Create a trainer for training the model
    meta_trainer = Trainer(args=meta_learning_args, task=meta_learning_task, model=model,
                           criterion=meta_learning_criterion, dummy_batch=dummy_batch, oom_batch=oom_batch)
    meta_epoch_itr = utils.create_epoch_iterator(task=meta_learning_task, dataset=train_dataset,
                                                 args=meta_learning_args, max_positions=max_positions)
    max_meta_epoch = meta_learning_args.max_epoch or math.inf  # inf default
    max_meta_update = meta_learning_args.max_update or math.inf  # inf default
    valid_subsets = meta_learning_args.valid_subset.split(',')
    for valid_subset in valid_subsets:
        # Load meta-dev split
        meta_learning_task.load_dataset(split=valid_subset)
    # Only rank 0 should attempt to create the required dir
    if meta_learning_args.distributed_rank == 0:
        os.makedirs(meta_learning_args.save_dir, exist_ok=True)
    return meta_epoch_itr, meta_trainer, max_meta_epoch, max_meta_update, valid_subsets


def combine_data(meta_train, fine_tune_args):
    combined_dataframe = pd.DataFrame()
    downstream_task = meta_train[0]
    for task in meta_train:
        user_data_frame = task.user_data_frame
        combined_dataframe = combined_dataframe.append(user_data_frame).reset_index(drop=True)
    rng = np.random.RandomState(seed=fine_tune_args.seed)
    combined_dataframe = combined_dataframe.sample(frac=1, random_state=rng).reset_index(drop=True)
    combined_dataframe.loc[combined_dataframe.split.str.startswith(fine_tune_args.train_subset), 'split'] = fine_tune_args.train_subset
    combined_dataframe.loc[combined_dataframe.split.str.startswith(fine_tune_args.valid_subset), 'split'] = fine_tune_args.valid_subset
    combined_dataframe.loc[combined_dataframe.split.str.startswith(fine_tune_args.test_subset), 'split'] = fine_tune_args.test_subset
    combined_fairseq_task = tasks.setup_task(args=fine_tune_args, user_data_frame=combined_dataframe,
                                             src_dict=downstream_task.src_dict, tgt_dict=downstream_task.tgt_dict)
    return combined_fairseq_task


def baseline(model, meta_learning_task, meta_learning_args, meta_learning_criterion, fine_tune_args):
    meta_epoch_itr, meta_trainer, max_meta_epoch, max_meta_update, valid_subsets = prepare_meta_task(
        model=model, meta_learning_task=meta_learning_task, meta_learning_args=meta_learning_args,
        meta_learning_criterion=meta_learning_criterion)
    # Only rank 0 should attempt to create the required dir
    if meta_learning_args.distributed_rank == 0:
        os.makedirs(fine_tune_args.save_dir, exist_ok=True)
    # Combine and do fine-tuning on combined data
    meta_train = meta_learning_task.dataset(meta_learning_args.train_subset)
    combined_fairseq_task = combine_data(meta_train=meta_train, fine_tune_args=fine_tune_args)
    # Fine-tune using the combined task
    criterion = combined_fairseq_task.build_criterion(fine_tune_args)
    utils.sgd(task=combined_fairseq_task, args=fine_tune_args, model=model, criterion=criterion,
              train_function=utils.train)
    # Evaluate on validation split
    maybe_validate(meta_epoch_itr=meta_epoch_itr, meta_learning_args=meta_learning_args, meta_trainer=meta_trainer,
                   meta_learning_task=meta_learning_task, valid_subsets=valid_subsets)

# Same as the baseline function, but also does intermediate meta-evaluation between updates
def baseline_with_meta_evaluation(model, meta_learning_task, meta_learning_args, meta_learning_criterion,
                                  fine_tune_args):
    meta_epoch_itr, meta_trainer, max_meta_epoch, max_meta_update, valid_subsets = prepare_meta_task(
        model=model, meta_learning_task=meta_learning_task, meta_learning_args=meta_learning_args,
        meta_learning_criterion=meta_learning_criterion)
    # Combine and do fine-tuning on combined data
    meta_train = meta_learning_task.dataset(meta_learning_args.train_subset)
    combined_fairseq_task = combine_data(meta_train=meta_train, fine_tune_args=fine_tune_args)
    # Fine-tune using the combined task
    criterion = combined_fairseq_task.build_criterion(fine_tune_args)
    import math
    from fairseq.trainer import Trainer
    combined_fairseq_task.load_dataset(fine_tune_args.train_subset)
    train_dataset = combined_fairseq_task.dataset(fine_tune_args.train_subset)
    # Make a dummy batch to (i) warm the caching allocator and (ii) as a  placeholder DistributedDataParallel when
    # there's an uneven number of batches per worker.
    max_positions = utils.resolve_max_positions(combined_fairseq_task.max_positions(), model.max_positions(),)
    dummy_batch = train_dataset.get_dummy_batch(num_tokens=fine_tune_args.max_tokens, max_positions=max_positions)
    oom_batch = combined_fairseq_task.dataset(fine_tune_args.train_subset).get_dummy_batch(1, max_positions)
    # Create a trainer for training the model
    trainer = Trainer(fine_tune_args, combined_fairseq_task, model, criterion, dummy_batch, oom_batch)
    epoch_itr = utils.create_epoch_iterator(task=combined_fairseq_task, dataset=train_dataset, args=fine_tune_args, max_positions=max_positions)
    max_epoch = fine_tune_args.max_epoch or math.inf
    max_update = fine_tune_args.max_update or math.inf
    # Do SGD on this task
    valid_subsets = fine_tune_args.valid_subset.split(',')
    lr = trainer.get_lr()
    batch_info = []
    # Always validate once before training
    valid_losses, _ = utils.validate(fine_tune_args, trainer, combined_fairseq_task, epoch_itr, valid_subsets)
    while lr > fine_tune_args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # Train the model for one epoch
        import collections
        import math
        from fairseq.data import iterators
        from fairseq import progress_bar
        from fairseq.meters import AverageMeter, ConcatentateMeter, BleuMeter
        """Train the model for one epoch."""
        # Update parameters every N batches
        update_freq = fine_tune_args.update_freq[epoch_itr.epoch - 1] \
            if epoch_itr.epoch <= len(fine_tune_args.update_freq) else fine_tune_args.update_freq[-1]

        # Initialize data iterator
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=fine_tune_args.fix_batches_to_gpus,
            shuffle=(epoch_itr.epoch >= fine_tune_args.curriculum),
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        progress = progress_bar.build_progress_bar(
            fine_tune_args, itr, epoch_itr.epoch, no_progress_bar='simple',
        )

        extra_meters = collections.defaultdict(lambda: AverageMeter())
        extra_meters['strings'] = ConcatentateMeter()
        extra_meters['bleu_stats'] = BleuMeter()

        valid_subsets = fine_tune_args.valid_subset.split(',')
        max_update = fine_tune_args.max_update or math.inf
        for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
            log_output = trainer.train_step(samples)
            if log_output is None:
                continue

            # log mid-epoch stats
            stats = utils.get_training_stats(trainer)
            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue  # these are already logged above
                if 'loss' in k:
                    extra_meters[k].update(v, log_output['sample_size'])
                else:
                    extra_meters[k].update(v)
                stats[k] = extra_meters[k].avg
            progress.log(stats, tag=fine_tune_args.train_subset, step=stats['num_updates'])

            # ignore the first mini-batch in words-per-second calculation
            if i == 0:
                trainer.get_meter('wps').reset()

            num_updates = trainer.get_num_updates()
            if fine_tune_args.save_interval_updates > 0 and num_updates % fine_tune_args.save_interval_updates == 0 and num_updates > 0:
                valid_losses, _ = utils.validate(fine_tune_args, trainer, combined_fairseq_task, epoch_itr, valid_subsets, train_progress=progress)
                utils.save_checkpoint(fine_tune_args, trainer, epoch_itr, valid_losses[0])

            if num_updates >= max_update:
                break

        # log end-of-epoch stats
        stats = utils.get_training_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
            stats[k+'_std'] = meter.std
        progress.print(stats, tag=fine_tune_args.train_subset, step=stats['num_updates'])

        # reset training meters
        for k in [
            'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
        ]:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        # Evaluate on validation split
        if epoch_itr.epoch % fine_tune_args.validate_interval == 0:
            valid_losses, _ = utils.validate(fine_tune_args, trainer, combined_fairseq_task, epoch_itr, valid_subsets)
        # save checkpoint
        if epoch_itr.epoch % fine_tune_args.save_interval == 0:
            utils.save_checkpoint(fine_tune_args, trainer, epoch_itr, valid_losses[0])
        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
    if batch_info is None:
        # Handle the original train function
        batch_info = []
    # Evaluate on validation split
    maybe_validate(meta_epoch_itr=meta_epoch_itr, meta_learning_args=meta_learning_args, meta_trainer=meta_trainer,
                   meta_learning_task=meta_learning_task, valid_subsets=valid_subsets)

def fairseq_reptile(model, meta_learning_task, meta_learning_args, meta_learning_criterion, fine_tune_args):
    meta_epoch_itr, meta_trainer, max_meta_epoch, max_meta_update, valid_subsets = prepare_meta_task(
        model=model, meta_learning_task=meta_learning_task, meta_learning_args=meta_learning_args,
        meta_learning_criterion=meta_learning_criterion)
    lr = meta_trainer.get_lr()
    # Evaluate on validation split
    print("| [Meta-Train Epoch] First validation ")
    maybe_validate(meta_epoch_itr=meta_epoch_itr, meta_learning_args=meta_learning_args, meta_trainer=meta_trainer, meta_learning_task=meta_learning_task, valid_subsets=valid_subsets)
    while lr > meta_learning_args.min_lr and meta_epoch_itr.epoch < max_meta_epoch and meta_trainer.get_num_updates() < max_meta_update:
        # Train the model for one epoch
        print("|[Meta-Train Epoch] ",meta_epoch_itr.epoch)
        utils.train(args=meta_learning_args, trainer=meta_trainer, task=meta_learning_task, epoch_itr=meta_epoch_itr, is_curriculum=meta_learning_args.is_curriculum)
        # Evaluate on validation split
        print("| [Meta-Train Epoch] validation start")
        valid_losses, _ = maybe_validate(meta_epoch_itr=meta_epoch_itr, meta_learning_args=meta_learning_args,
                                         meta_trainer=meta_trainer, meta_learning_task=meta_learning_task,
                                         valid_subsets=valid_subsets)
        # save checkpoint
        if meta_epoch_itr.epoch % meta_learning_args.save_interval == 0:
            utils.save_checkpoint(meta_learning_args, meta_trainer, meta_epoch_itr, valid_losses[0])
        # only use first validation loss to update the learning rate
        lr = meta_trainer.lr_step(meta_epoch_itr.epoch, valid_losses[0])
        print("|[Meta-Train Epoch END] ",meta_epoch_itr.epoch)

# Meta CL utils
def cdf_func(scores_arr, smooth_tolorance=5, cdf_bins='auto'):
    counts, bin_edges = np.histogram(scores_arr[:len(scores_arr)-5], bins=cdf_bins, normed=True)
    cdf = np.cumsum(counts)
    boundary = bin_edges[1:]
    cdf_prob = cdf/cdf[-1]
    boundary = np.append(boundary, scores_arr[-1])
    cdf_prob[-1] -= 1e-05
    cdf_prob = np.append(cdf_prob, 1.0)
    cdf_val = dict(zip(cdf_prob, boundary))
    return cdf_val

def model_competence_frac(t, T, c0=0.1):
    tmp = t * ((1 - c0 * c0) / T) + c0 * c0
    c_square = tmp ** 0.5
    if c_square > 0.97:
        tmp = 1.0
    competence_prob = min(1.0, c_square)
    return competence_prob

def log_frac(t, T):
    tmp = np.log(t+1)/np.log(T)
    if tmp > 0.97:
        tmp = 1.0
    return min(1.0, tmp)

def ladder(t, T, stage=3):
    K = T/stage
    tmp = math.ceil((t+1)/K)*K/T
    return min(1.0, tmp)

def modify_trainer(meta_learning_args, full_meta_learning_task, meta_trainer, frac_type, cur_step, max_step):
    frac = None
    if frac_type == 'log':
        frac = log_frac(cur_step+1, max_step)
    elif frac_type == 'competence':
        frac = model_competence_frac(cur_step, max_step, 0.1)
    elif frac_type == 'ladder':
        frac = ladder(cur_step, max_step)
    assert (frac is not None)
    # task_scores = [i.task_score for i in meta_learning_task.datasets[meta_learning_args.train_subset].meta_tasks]
    # Only update trainer-train_subset & epoch iter
    full_train_subset = full_meta_learning_task.meta_train_tasks
    sample_len = min(int(math.ceil(frac * len(full_train_subset))), len(full_train_subset))
    train_data_set_frac = full_train_subset[:sample_len]
    meta_trainer.task.datasets[meta_learning_args.train_subset].meta_tasks = train_data_set_frac
    meta_learning_task = meta_trainer.task
    # meta_trainer = Trainer(args=meta_learning_args, task=meta_learning_task, model=model,
    #                        criterion=meta_learning_criterion, dummy_batch=dummy_batch, oom_batch=oom_batch)
    # train_dataset = meta_trainer.datasets[meta_learning_args.train_subset]
    max_positions = utils.resolve_max_positions(meta_learning_task.max_positions(), meta_trainer.model.max_positions(),)
    meta_epoch_itr = utils.create_epoch_iterator(task=meta_learning_task, dataset=meta_learning_task.dataset(meta_learning_args.train_subset),args=meta_learning_args, max_positions=max_positions)
    return meta_trainer, meta_epoch_itr, meta_learning_task

def compet_meta_cl(model, meta_learning_task, meta_learning_args, meta_learning_criterion, fine_tune_args):
    meta_epoch_itr, meta_trainer, max_meta_epoch, max_meta_update, valid_subsets = prepare_meta_task(
        model=model, meta_learning_task=meta_learning_task, meta_learning_args=meta_learning_args,
        meta_learning_criterion=meta_learning_criterion)
    full_meta_learning_task = copy.deepcopy(meta_learning_task)
    frac_type = meta_learning_args.cl_frac
    assert (frac_type is not None)
    lr = meta_trainer.get_lr()
    # Evaluate on validation split
    print("| [Meta-Train Epoch] First validation ")
    maybe_validate(meta_epoch_itr=meta_epoch_itr, meta_learning_args=meta_learning_args, meta_trainer=meta_trainer, meta_learning_task=meta_learning_task, valid_subsets=valid_subsets)
    while lr > meta_learning_args.min_lr and meta_epoch_itr.epoch < max_meta_epoch and meta_trainer.get_num_updates() < max_meta_update:
        # Train the model for one epoch
        last_epoch = int(meta_epoch_itr.epoch)
        meta_trainer, meta_epoch_itr, meta_learning_task = modify_trainer(meta_learning_args, full_meta_learning_task, meta_trainer, frac_type, meta_trainer.get_num_updates(), max_meta_update)
        meta_epoch_itr.epoch = last_epoch
        print('|[Meta-Train Epoch] {} Cur step: {}/{}, task_num: {}'.format(
            meta_epoch_itr.epoch,
            meta_trainer.get_num_updates(),
            max_meta_update,
            len(meta_learning_task.dataset(meta_learning_args.train_subset).meta_tasks)
        )
        )
        utils.train(args=meta_learning_args, trainer=meta_trainer, task=meta_learning_task, epoch_itr=meta_epoch_itr, is_curriculum=meta_learning_args.is_curriculum)
        # Evaluate on validation split
        print("| [Meta-Train Epoch] validation start")
        valid_losses, _ = maybe_validate(meta_epoch_itr=meta_epoch_itr, meta_learning_args=meta_learning_args,
                                         meta_trainer=meta_trainer, meta_learning_task=meta_learning_task,
                                         valid_subsets=valid_subsets)
        # save checkpoint
        if meta_epoch_itr.epoch % meta_learning_args.save_interval == 0:
            utils.save_checkpoint(meta_learning_args, meta_trainer, meta_epoch_itr, valid_losses[0])
        # only use first validation loss to update the learning rate
        lr = meta_trainer.lr_step(meta_epoch_itr.epoch, valid_losses[0])
        print("| [Meta-Train Epoch END] ")

def distributed_main(i, meta_learning_args, downstream_args, fine_tune_args, main_fn):
    meta_learning_args.device_id = i
    if meta_learning_args.distributed_rank is None:  # torch.multiprocessing.spawn
        meta_learning_args.distributed_rank = i
    downstream_args.device_id = i
    if downstream_args.distributed_rank is None:  # torch.multiprocessing.spawn
        downstream_args.distributed_rank = i
    if fine_tune_args is not None and fine_tune_args.distributed_rank is None:  # torch.multiprocessing.spawn
        fine_tune_args.distributed_rank = i
    main_fn(meta_learning_args=meta_learning_args, downstream_args=downstream_args, fine_tune_args=fine_tune_args)


def get_checkpoint_path(root, restore_file):
    if os.path.isabs(restore_file):
        checkpoint_path = restore_file
    else:
        checkpoint_path = os.path.join(root, restore_file)
    return checkpoint_path


def load_model(root, restore_file, task):
    model_file = get_checkpoint_path(root=root, restore_file=restore_file)
    model_ensemble, model_args = load_model_ensemble(filenames=[model_file], task=task)
    assert (len(model_ensemble) == 1)
    model = model_ensemble[0]
    return model, model_args


def cli_main(main_fn):
    argv = sys.argv[1:]
    # This is a maker that separates meta-learning arguments from downstream training arguments
    split_index = argv.index('---')
    meta_argv = argv[:split_index]
    maybe_downstream_argv = argv[split_index+1:]
    parser = options.get_meta_training_parser()
    meta_learning_args = options.parse_args_and_arch(parser, input_args=meta_argv)
    fine_tune_args = None
    if meta_learning_args.baseline:
        split_index = maybe_downstream_argv.index('---')
        downstream_argv = maybe_downstream_argv[:split_index]
        baseline_argv = maybe_downstream_argv[split_index+1:]
        parser = options.get_meta_learning_parser()
        fine_tune_args = options.parse_args_and_arch(parser, input_args=baseline_argv)
    else:
        downstream_argv = maybe_downstream_argv
    parser = options.get_meta_learning_parser()
    downstream_args = options.parse_args_and_arch(parser, input_args=downstream_argv)
    print('Meta-learning Arguments: ')
    print(meta_learning_args)
    print('Downstream Arguments: ')
    print(downstream_args)
    print('Fine-tune Args: ')
    print(fine_tune_args)
    if meta_learning_args.distributed_init_method is None:
        distributed_utils.infer_init_method(meta_learning_args)

    if meta_learning_args.distributed_init_method is not None:
        # distributed training
        distributed_main(meta_learning_args.device_id, meta_learning_args=meta_learning_args,
                         downstream_args=downstream_args, fine_tune_args=fine_tune_args, main_fn=main_fn)
    elif meta_learning_args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        port = random.randint(10000, 20000)
        meta_learning_args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        meta_learning_args.distributed_rank = None  # set based on device id
        if max(meta_learning_args.update_freq) > 1 and meta_learning_args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(meta_learning_args, downstream_args, fine_tune_args, main_fn),
            nprocs=meta_learning_args.distributed_world_size,
        )
    else:
        # single GPU training
        main_fn(meta_learning_args=meta_learning_args, downstream_args=downstream_args, fine_tune_args=fine_tune_args)


def do_plot(plot, task, args, batch_info):
    if plot:
        import matplotlib.pyplot as plt
        plt.cla()
        train_dataset = task.dataset(args.train_subset)
        x_train = train_dataset.x
        y_train = train_dataset.y
        for inner_iter, predict_output in enumerate(batch_info):
            frac = (inner_iter + 1) / len(batch_info)
            x = [p[0] for p in predict_output]
            y = [p[1] for p in predict_output]
            order = np.argsort(x)
            x = [x[i] for i in order]
            y = [y[i] for i in order]
            plt.plot(x, y, label='pred after %i' % inner_iter, color=(frac, 0, 1-frac))
        # Plot the ground truth
        plt.plot(x_train, y_train, label='true', color=(0, 1, 0))
        plt.plot(x_train, y_train, 'x', label='train', color='k')
        plt.ylim(-4, 4)
        plt.legend(loc='lower right')
        plt.show()


def evaluate_all_tasks(all_tasks, model, criterion, args, plot):
    number_of_tasks = len(all_tasks)
    all_loss_values = np.zeros(number_of_tasks)
    all_bleu = np.zeros(number_of_tasks)
    for i, task in enumerate(all_tasks):
        all_loss_values[i], all_bleu[i], batch_info = utils.fine_tune_on_task(model=model, task=task, args=args)
        do_plot(plot=plot, task=task, args=args, batch_info=batch_info)
    return all_loss_values, all_loss_values.mean(), all_loss_values.std(), all_bleu, all_bleu.mean(), all_bleu.std()


def build_reptile_function(is_baseline, is_curriculum=False):
    if is_baseline:
        return baseline
    elif is_curriculum:
        return fairseq_reptile
    else:
        return fairseq_reptile


def run_maybe_distributed_reptile(meta_learning_args, downstream_args, load_meta_tasks_fn, fine_tune_args):
    seed = downstream_args.seed
    if torch.cuda.is_available() and not meta_learning_args.cpu:
        torch.cuda.set_device(meta_learning_args.device_id)
    torch.manual_seed(seed)
    meta_train_tasks, meta_dev_tasks = load_meta_tasks_fn()
    # build model and criterion
    print('| training on {} GPUs'.format(meta_learning_args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(meta_learning_args.max_tokens, meta_learning_args.max_sentences,))
    # Reptile training loop
    print('setup for meta-learning task')
    # Meta-learning task list
    meta_learning_task = tasks.setup_task(args=meta_learning_args, meta_train_tasks=meta_train_tasks,
                                          meta_dev_tasks=meta_dev_tasks, meta_test_tasks=None)
    print('building meta-learning model...')
    model = meta_learning_task.build_model(meta_learning_args)  # Transformer RAW
    state = load_checkpoint_to_cpu(meta_learning_args.restore_file)
    model.load_state_dict(state['model'], strict=False)
    meta_learning_criterion = meta_learning_task.build_criterion(meta_learning_args)  # MAML, FoMAML
    print(model)
    print('| model {}, criterion {}'.format(meta_learning_args.arch, meta_learning_criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),))
    reptile_function = build_reptile_function(is_baseline=meta_learning_args.baseline, is_curriculum=meta_learning_args.is_curriculum)
    reptile_function(model=model, meta_learning_task=meta_learning_task, meta_learning_args=meta_learning_args,
                     meta_learning_criterion=meta_learning_criterion, fine_tune_args=fine_tune_args)
