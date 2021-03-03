import logging
import os
import math
import sys
import torch
import random
import pathlib
import ipdb
from pandarallel import pandarallel
import sentencepiece as spm
import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import deepcopy
from functools import partial
from itertools import cycle, islice
from collections import namedtuple, OrderedDict
import fairseq
from fairseq.utils import run_inference, sgd, bucket_sgd, validate
from fairseq import meta_learning_utils
from fairseq import options, distributed_utils, tasks
from fairseq.data import Dictionary

# DomainTaskData = namedtuple('DomainTaskData', 'meta_train meta_dev meta_test')
logging.basicConfig(level=logging.WARNING)
MetaSplitData = namedtuple('MetaSplitData', 'DOMAIN support query dev')
SEED = 7
SPM_MODEL_PATH = '/home/zhanrunzhe/data/domain/mixed/joint.40k.model'
SPM_PROCESSER = spm.SentencePieceProcessor()
SPM_PROCESSER.Load(SPM_MODEL_PATH)
pandarallel.initialize()

# To get a model, we need a task, we need an easy abstraction to create a task! The annoying way is that fairseq doesn't make this easy enough, the only way to create a task is to call tasks.setup_task, which need args, not too bad though!
# To construct a task, we also need the dictionary!

# TODO can also use argparse from fairseq

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

def get_task_list_sting(task_list):
    task_list_string = ''
    for task in task_list:
        task_list_string += ' ' + str(task.user_data_frame.task_group.any().split('.')[-1])
    return task_list_string

def apply_roundrobin(domains, tasks_by_domain, cl_task_reorder=False, train_limit=100000, valid_limit=None, test_limit=None):
    all_meta_train_tasks = []
    # all_meta_dev_tasks = []
    # all_meta_test_tasks = []
    # we want a balanced set of different domains, so that when we merge for the double fine-tune baseline, we can win!
    for domain in domains:
        domain_task_data = tasks_by_domain[domain]
        meta_train = domain_task_data
        # meta_dev = domain_task_data.meta_dev
        # meta_test = domain_task_data.meta_test
        all_meta_train_tasks.append(meta_train)
        # all_meta_dev_tasks.append(meta_dev)
        # all_meta_test_tasks.append(meta_test)
    assert len(all_meta_train_tasks) == len(domains)
    print('size of domain: ', len(all_meta_train_tasks))
    balanced_meta_train = list(islice(roundrobin(*all_meta_train_tasks), train_limit))
    if not cl_task_reorder:
        random.shuffle(balanced_meta_train)
    else:
        print('Orignal meta-training on: ', get_task_list_sting(balanced_meta_train))
        print('Task Score:', [i.task_score for i in balanced_meta_train])
        balanced_meta_train = sorted(balanced_meta_train, key=lambda x: x.task_score)
    # ipdb.set_trace()
    print('Meta-training on: ', get_task_list_sting(balanced_meta_train))
    print('Task Score:', [i.task_score for i in balanced_meta_train])
    print('Total meta-training tasks: ', len(balanced_meta_train))
    # balanced_meta_test = list(islice(roundrobin(*all_meta_test_tasks), test_limit))
    # task_score_dict = read_task_score_file('task_score')
    # print('size of balanced meta-train: ', len(balanced_meta_train))
    # print('training on: ', get_task_list_sting(balanced_meta_train))
    # reorder_meta_train = sorted(balanced_meta_train, key=lambda x:task_score_dict[x.user_data_frame.user.any().strip()])
    # print('curriculum training on: ', get_task_list_sting(reorder_meta_train))
    # print('size of balanced meta-dev: ', len(balanced_meta_dev))
    # print('validating on: ', get_task_list_sting(balanced_meta_dev))
    # print('size of balanced meta-test: ', len(balanced_meta_test))
    # print('testing on: ', get_task_list_sting(balanced_meta_test))
    return balanced_meta_train  #  , balanced_meta_dev, balanced_meta_test

# Converts user translation task data to fairseq task
def to_fairseq_task(task, task_score, args, src_dict, tgt_dict):
    # task is a Tuple, first entry is the key
    task_args = deepcopy(args)
    domain_task = task.task_group.any().split('.')[-1].split('_')[0]
    task_args.train_subset = task_args.train_subset + '.' + domain_task
    task_args.valid_subset = task_args.valid_subset + '.' + domain_task
    task_args.test_subset = task_args.test_subset + '.' + domain_task
    fairseq_task = tasks.setup_task(args=task_args, user_data_frame=task, src_dict=src_dict, tgt_dict=tgt_dict,task_score=task_score)
    # ipdb.set_trace() 
    return fairseq_task

def get_info_table(data_dir, split_subset, source_lang, target_lang, with_score=False):
    source_reader = open(os.path.join(data_dir, split_subset + '.' + source_lang),'r', encoding='utf8')
    target_reader = open(os.path.join(data_dir, split_subset + '.' + target_lang),'r', encoding='utf8')
    source_sents = [line.strip() for line in source_reader.readlines()]
    target_sents = [line.strip() for line in target_reader.readlines()]
    source_reader.close()
    target_reader.close()
    if with_score:
        score_reader = open(os.path.join(data_dir, split_subset + '.score'), 'r', encoding='utf8')
        sents_score = [abs(float(sc)) for sc in score_reader.readlines()]
        if len(sents_score) == 0:
            logging.warning('No Score for this domain !')
            sents_score = [0.0] * len(source_sents)
        score_reader.close()
        info_df = pd.DataFrame({
            'src': source_sents,
            'tgt': target_sents,
            'score': sents_score,
            'task_group': ['TBD' for i in range(0, len(source_sents))]
        })
    else:
        info_df = pd.DataFrame({
            'src': source_sents,
            'tgt': target_sents,
            'task_group': ['TBD' for i in range(0, len(source_sents))]
        })
    return info_df


def form_split_sents_df(data_dir, support_subset, query_subset, source_lang, target_lang, domain, dev_subset=None):
    if support_subset == None and query_subset == None and dev_subset != None:
        dev_sentences = get_info_table(data_dir=data_dir, split_subset=dev_subset, source_lang=source_lang,target_lang=target_lang, with_score=True)
        return MetaSplitData(DOMAIN=domain, support=None, query=None, dev=dev_sentences)
    else:
        support_sentences = get_info_table(data_dir=data_dir, split_subset=support_subset, source_lang=source_lang,target_lang=target_lang, with_score=True)
        query_sentences = get_info_table(data_dir=data_dir, split_subset=query_subset, source_lang=source_lang,target_lang=target_lang, with_score=True)
        meta_split_sents = MetaSplitData(DOMAIN=domain, support=support_sentences, query=query_sentences, dev=None)
        return meta_split_sents

# TODO send limit to read only what we need
def load_static_meta_splits(root, domain, meta_dev_subset, meta_test_subset, downstream_train_subset, downstream_valid_subset, downstream_test_subset):
    meta_dev_dir = os.path.join(root, domain, 'meta_split', meta_dev_subset)
    meta_test_dir = os.path.join(root, domain, 'meta_split', meta_test_subset)
    # meta_test_dir = root + '/' + domain + '_de-en/' + meta_test_subset
    source_lang = 'en'
    target_lang = 'de'
    train_subset = downstream_train_subset
    valid_subset = downstream_valid_subset
    test_subset = downstream_test_subset
    meta_test = form_split_sents_df(data_dir=meta_test_dir, support_subset=train_subset,query_subset=test_subset, source_lang=source_lang, target_lang=target_lang, domain=domain)
    meta_dev = form_split_sents_df(data_dir=meta_dev_dir, support_subset=None,
                                   query_subset=None, source_lang=source_lang, target_lang=target_lang,domain=domain, dev_subset=valid_subset)
    # meta_test = load_single_meta_split(split_directory=meta_test_dir)
    return meta_test, meta_dev

def fisher_breaks_divider_fn(scores_array, num_bucket):
    pass

def equal_divider_fn(scores_array, num_bucket):
    task_group_idx = []
    scores_array = np.array(scores_array)
    splited_group = np.array_split(scores_array, num_bucket)
    for idx, group in enumerate(splited_group):
        task_group_idx += len(group) * [idx]
    return task_group_idx

def assign_group_by_bucket(signature, bucket, bucket_divider, info_df, cl_assign=False):
    if cl_assign:
        info_df = info_df.sort_values(by='score')
    # percentage curriculum
    task_group_idx = bucket_divider(info_df.score, bucket)
    info_df.task_group = [signature + '_B' + str(i) for i in task_group_idx]
    logging.info('[{}] buckets_num: {}'.format(signature,bucket))
    return info_df, task_group_idx

def split_fine_tune_domain_task(split_sents_df, bucket, domain, cl_sample=False):
    task_df_list = []
    task_df_score_list = []
    group_support_df, bucket_idx = assign_group_by_bucket(
        signature='support.' + domain,
        bucket=bucket,
        bucket_divider=equal_divider_fn,
        info_df=split_sents_df.support,
        cl_assign=cl_sample
    )
    print("buckets score mean: {}".format(group_support_df.groupby('task_group').score.mean().to_list()))
    ft_with_bucket = group_support_df
    return ft_with_bucket

def merge_meta_train(meta_train_df_list):
    support_df_merge = pd.DataFrame({
        'src': [],
        'tgt': [],
        'score': [],
        'task_group': []
    })
    query_df_merge = pd.DataFrame({
        'src': [],
        'tgt': [],
        'score': [],
        'task_group': []
    })
    for i in meta_train_df_list:
        support_df_merge = support_df_merge.append(i.support)
        query_df_merge = query_df_merge.append(i.query)
    return MetaSplitData(DOMAIN='CL', support=support_df_merge.reset_index(drop=True), query=query_df_merge.reset_index(drop=True), dev=None)


def read_tasks_ordered_by_domain(domains, source_dictionary, target_dictionary, args, meta_dev_subset, meta_test_subset, root, downstream_train_subset,downstream_valid_subset, downstream_test_subset, cl_learn):
    tasks_by_domain = OrderedDict()
    dev_by_domain = OrderedDict()
    # rng = np.random.RandomState(seed)
    for domain in domains:
        meta_test, meta_dev = load_static_meta_splits(root=root, domain=domain,
                                                      meta_dev_subset=meta_dev_subset,
                                                      meta_test_subset=meta_test_subset,
                                                      downstream_train_subset=downstream_train_subset,
                                                      downstream_valid_subset=downstream_valid_subset,
                                                      downstream_test_subset=downstream_test_subset)
        # Split into task (store in Dataframe) list
        meta_dev.dev.task_group = downstream_valid_subset + '.' + domain
        meta_test.query.task_group = downstream_test_subset + '.' + domain
        ft_support_with_buckets = split_fine_tune_domain_task(meta_test, bucket=3, domain=domain, cl_sample=cl_learn)
        ft_data_split = ft_support_with_buckets.append(meta_test.query)
        ft_data_split = ft_data_split.append(meta_dev.dev)
        # Convert data frames to fairseq tasks
        convert_to_fairseq_task_fn = partial(to_fairseq_task, args=args, src_dict=source_dictionary, tgt_dict=target_dictionary)
        # TODO: Pause Here
        ft_task = list(map(convert_to_fairseq_task_fn, [ft_data_split], [ft_support_with_buckets.score]))
        tasks_by_domain[domain] = ft_task
        # dev_by_domain[domain] = meta_dev
        print('[{}] Number of meta-test tasks: {} '.format(domain, len(ft_task)))
        print('[{}] Number of finetune sentences: {} '.format(domain, len(ft_support_with_buckets)))
        print('[{}] Number of tests sentences: {} '.format(domain, len(meta_test.query)))
        print('[{}] Number of dev sentences: {} '.format(domain, len(meta_dev.dev)))
        # print('Length of meta-dev: ', len(meta_dev))
    print('Total number of tasks: ', sum([len(i) for i in tasks_by_domain.values()]))
    # ipdb.set_trace()
    return tasks_by_domain


# ===================================================================

def prepare_meta_adaptation_tasks(meta_learning_args, downstream_args):
    domains = meta_learning_args.domains
    print('domains:')
    print(domains)
    if len(domains) < 1:
        raise "Domain insufficient!"
    assert len(meta_learning_args.data) == 1
    root = meta_learning_args.data[0]
    source_dict = os.path.join(root, 'dict.en.txt')
    target_dict = os.path.join(root, 'dict.de.txt')
    source_dictionary = Dictionary.load(source_dict)
    target_dictionary = Dictionary.load(target_dict)
    tasks_by_domain = read_tasks_ordered_by_domain(domains=domains,
                                                   source_dictionary=source_dictionary,
                                                   target_dictionary=target_dictionary,
                                                   args=downstream_args,
                                                   meta_dev_subset=meta_learning_args.valid_subset,
                                                   meta_test_subset=meta_learning_args.test_subset,
                                                   root=root,
                                                   downstream_train_subset=downstream_args.train_subset,
                                                   downstream_valid_subset=downstream_args.valid_subset,
                                                   downstream_test_subset=downstream_args.test_subset,
                                                   cl_learn=meta_learning_args.is_curriculum)
    # all_meta_train = apply_roundrobin(domains, tasks_by_domain, cl_task_reorder=meta_learning_args.is_curriculum)
    # all_dev_task = []
    # for domain in domains:
        # all_dev_task += dev_by_domain[domain]
    return tasks_by_domain


def fine_tune_on_task(model, task, args):
    from copy import deepcopy
    from fairseq.utils import train
    # save snapshot before evaluation
    weights_before = deepcopy(model.state_dict())
    # train on meta-test task
    criterion = task.build_criterion(args)
    sgd_dict = sgd(task=task, args=args, model=model, criterion=criterion, train_function=train)
    loss_val, _ = validate(args=args, trainer=sgd_dict['trainer'], task=task, epoch_itr=sgd_dict['epoch_itr'],
                           subsets=args.valid_subset.split(','))
    # _, bleu = run_inference(args=args, task=task, model=model, subset=args.valid_subset)
    # restore from snapshot
    model.load_state_dict(weights_before)
    return loss_val, sgd_dict['batch_info']


def cl_fine_tune_on_task(model, task, args):
    from copy import deepcopy
    from fairseq.utils import train
    # save snapshot before evaluation
    weights_before = deepcopy(model.state_dict())
    # train on meta-test task
    criterion = task.build_criterion(args)
    max_epoch = args.max_epoch or math.inf
    cur_epoch = 0
    while True:
        if cur_epoch > args.max_epoch:
            break
        for idx, bucket in enumerate(task.datasets[args.train_subset+'_bucket']):
            if cur_epoch == 0:
                sgd_dict = bucket_sgd(task=task, args=args, model=model, criterion=criterion, bucket=idx, cur_epoch=cur_epoch,train_function=train)
            else:
                sgd_dict = bucket_sgd(task=task, args=args, model=model, criterion=criterion, bucket=idx, cur_epoch=cur_epoch,train_function=train, last_trainer=sgd_dict['trainer'])
            cur_epoch = sgd_dict['epoch_itr'].epoch
            if cur_epoch > args.max_epoch:
                break
            loss_val, _ = validate(args=args, trainer=sgd_dict['trainer'], task=task, epoch_itr=sgd_dict['epoch_itr'], subsets=args.valid_subset.split(','))
    # _, bleu = run_inference(args=args, task=task, model=model, subset=args.test_subset)
    # restore from snapshot
    model.load_state_dict(weights_before)
    return loss_val, sgd_dict['batch_info']


def run_fine_tuning_evaluation(tasks, model, args):
    # TODO make sure command line tool and python interface gives the same results
    max_n_tasks = sum([len(i) for i in tasks.values()])
    bleu_scores_before = np.zeros(max_n_tasks)
    bleu_scores_after = np.zeros(max_n_tasks)
    root_path = args.save_dir
    for i, fairseq_task_key in enumerate(tqdm(tasks.keys())):
        print('domain for this task: ', fairseq_task_key)
        # we now want to make sure fine-tuning works for these clusters, so let's test that
        # TODO design a better experiment
        fairseq_task = tasks[fairseq_task_key]
        assert (len(fairseq_task) == 1)
        fairseq_task = fairseq_task[0]
        fairseq_task.load_dataset(args.test_subset, fine_tune=True)
        # fairseq_task.load_dataset(args.train_subset, fine_tune=True)
        fairseq_task.load_dataset(args.train_subset, fine_tune=True, bucket=args.is_curriculum)
        fairseq_task.load_dataset(args.valid_subset, fine_tune=True)
        # print('computing BLEU score before fine-tuning...')
        # bleu, _ = run_inference(args, fairseq_task, model=model, subset=args.test_subset)
        # bleu_scores_before[i] = bleu
        # print('BLEU before: ', bleu)
        args.save_dir = os.path.join(root_path, fairseq_task_key)
        pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        # ipdb.set_trace()
        print('now fine-tuning...')
        if args.is_curriculum:
            _, _ = cl_fine_tune_on_task(model=model, task=fairseq_task, args=args)
        else:
            fine_tune_on_task(model=model, task=fairseq_task, args=args)
        # bleu_scores_after[i] = bleu
        # print('BLEU after: ', bleu)
        # print('median before: ', np.median(bleu_scores_before[:i + 1]), ' mean before: ',
        #       bleu_scores_before[:i + 1].mean(), ' sd: ', bleu_scores_before[:i + 1].std())
        # print('median after: ', np.median(bleu_scores_after[:i + 1]), ' mean  after: ',
        #       bleu_scores_after[:i + 1].mean(), ' sd: ', bleu_scores_after[:i + 1].std())
        print('done with this task')


def main_meta_adaptation(meta_learning_args, downstream_args):
    from fairseq.checkpoint_utils import load_model_ensemble, load_checkpoint_to_cpu
    tasks_by_domain = prepare_meta_adaptation_tasks(meta_learning_args=meta_learning_args, downstream_args=downstream_args)
    dummy_task = tasks.setup_task(args=meta_learning_args, meta_train_tasks=None, meta_dev_tasks=None, meta_test_tasks=None)
    # print('building meta-learning model...')
    # model = dummy_task.build_model(meta_learning_args)  # Transformer RAW
    # state = load_checkpoint_to_cpu(downstream_args.restore_file)
    # model.load_state_dict(state['model'], strict=False)
    # dummy_task = [i for i in tasks_by_domain.values()][0][0]
    model, _ = meta_learning_utils.load_model(root=meta_learning_args.data[0], restore_file=downstream_args.restore_file, task=dummy_task)
    del dummy_task
    run_fine_tuning_evaluation(tasks=tasks_by_domain, model=model, args=downstream_args)

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

def cli_main(main_fn):
    argv = sys.argv[1:]
    # This is a maker that separates meta-learning arguments from downstream training arguments
    split_index = argv.index('---')
    meta_argv = argv[:split_index]
    parser = options.get_meta_training_parser(default_task='meta-curriculum')
    meta_learning_args = options.parse_args_and_arch(parser, input_args=meta_argv)
    maybe_downstream_argv = argv[split_index+1:]
    parser = options.get_meta_finetune_parser()
    fine_tune_args = options.parse_args_and_arch(parser, input_args=maybe_downstream_argv)
    print('Meta-learning Arguments: ')
    print(meta_learning_args)
    print('Fine-tune Args: ')
    print(fine_tune_args)
    if meta_learning_args.distributed_init_method is None:
        distributed_utils.infer_init_method(meta_learning_args)

    if meta_learning_args.distributed_init_method is not None:
        # distributed training
        distributed_main(meta_learning_args.device_id, meta_learning_args=meta_learning_args,
                         downstream_args=fine_tune_args, fine_tune_args=fine_tune_args, main_fn=main_fn)
    elif meta_learning_args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        port = random.randint(10000, 20000)
        meta_learning_args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        meta_learning_args.distributed_rank = None  # set based on device id
        if max(meta_learning_args.update_freq) > 1 and meta_learning_args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(meta_learning_args, fine_tune_args, fine_tune_args, main_fn),
            nprocs=meta_learning_args.distributed_world_size,
        )
    else:
        # single GPU training
        main_fn(meta_learning_args=meta_learning_args, downstream_args=fine_tune_args)

if __name__ == "__main__":
    cli_main(main_fn=main_meta_adaptation)