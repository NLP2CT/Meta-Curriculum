import logging
import os
import math
import sys
import torch
import random
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
    domain_task = task.task_group.any().split('.')[-1]
    task_args.train_subset = task_args.train_subset + '.' + domain_task
    task_args.valid_subset = task_args.test_subset + '.' + domain_task
    # task_args.test_subset = task_args.test_subset + '.' + domain_task
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
def load_static_meta_splits(root, domain, meta_train_subset, meta_dev_subset, meta_test_subset, downstream_train_subset, downstream_valid_subset, downstream_test_subset):
    meta_train_dir = os.path.join(root, domain, 'meta_split', meta_train_subset)
    meta_dev_dir = os.path.join(root, domain, 'meta_split', meta_dev_subset)
    # meta_test_dir = root + '/' + domain + '_de-en/' + meta_test_subset
    source_lang = 'en'
    target_lang = 'de'
    train_subset = downstream_train_subset
    valid_subset = downstream_valid_subset
    test_subset = downstream_test_subset
    meta_train = form_split_sents_df(data_dir=meta_train_dir, support_subset=train_subset,query_subset=test_subset, source_lang=source_lang, target_lang=target_lang, domain=domain)
    meta_dev = form_split_sents_df(data_dir=meta_dev_dir, support_subset=None,
                                   query_subset=None, source_lang=source_lang, target_lang=target_lang,domain=domain, dev_subset=valid_subset)
    # meta_test = load_single_meta_split(split_directory=meta_test_dir)
    return meta_train, meta_dev

def assign_group_by_token(signature, task_tokens, info_df, cl_assign=False):
    cur_tokens = 0
    task_group_cnt = 0
    task_signature = []
    if cl_assign:
        info_df = info_df.sort_values(by='score')
    for idx, row in info_df.iterrows():
        if cur_tokens >= task_tokens:
            cur_tokens = 0
            task_group_cnt += 1
            sig = signature + '_' + str(task_group_cnt)
            task_signature.append(sig)
        else:
            # cur_tokens += len(row['src'].split())
            cur_tokens += len(SPM_PROCESSER.decode(row['src'].split()).split())
            sig = signature + '_' + str(task_group_cnt)
            task_signature.append(sig)
    info_df.task_group = task_signature
    logging.info('[{}] task_num: {}'.format(signature,str(task_group_cnt+1)))
    # logging.info('[{}] task_score: {}'.format(task_score))
    return info_df

def split_domain_task(split_sents_df, support_tokens, query_tokens, domain, cl_sample=False, is_dev=False):
    #  support.emea.0
    if is_dev:
        df = split_sents_df.dev
        df.task_group = ['support.' + domain + '_dev'] * 1000 + ['query.' + domain + '_dev'] * (len(df) - 1000)
        return [df]
    task_df_list = []
    task_df_score_list = []
    group_supoort_df = assign_group_by_token(
        signature='support.' + domain,
        task_tokens=support_tokens,
        info_df=split_sents_df.support,
        cl_assign=cl_sample
    )
    group_query_df = assign_group_by_token(
        signature='query.' + domain,
        task_tokens=query_tokens,
        info_df=split_sents_df.query,
        cl_assign=cl_sample
    )
    support_task_num = len(group_supoort_df.groupby('task_group'))
    query_task_num = len(group_query_df.groupby('task_group'))
    if support_task_num != support_task_num:
        logging.warn('Tasks may be unbalanced!')
        if abs(support_task_num - query_task_num) >= 3:
            raise "Extremely unbalanced support & query split, Terminated!"
    #  Merge to list
    for task_id in range(0, min(support_task_num, query_task_num)):
        s_label = 'support.' + domain + '_' + str(task_id)
        q_label = 'query.' + domain + '_' + str(task_id)
        task_df_list.append(group_supoort_df[group_supoort_df.task_group == s_label].append(group_query_df[group_query_df.task_group == q_label]).reset_index(drop=True))
    for idx,i in enumerate(task_df_list):
        task_df_score_list.append(i.score.mean())
    task_with_score = list(zip(task_df_list, task_df_score_list))
    # ipdb.set_trace()
    if cl_sample:
        task_with_score.sort(key=lambda x : x[1])
    # TODO: PAUSE
    # ipdb.set_trace()
    return task_with_score

def split_currciulum_task(split_sents_df, support_tokens, query_tokens, cl_sample=True):
    task_df_list = []
    task_df_score_list = []
    domain = 'CL'
    group_supoort_df = assign_group_by_token(
        signature='support.' + domain,
        task_tokens=support_tokens,
        info_df=split_sents_df.support,
        cl_assign=cl_sample
    )
    group_query_df = assign_group_by_token(
        signature='query.' + domain,
        task_tokens=query_tokens,
        info_df=split_sents_df.query,
        cl_assign=cl_sample
    )
    support_task_num = len(group_supoort_df.groupby('task_group'))
    query_task_num = len(group_query_df.groupby('task_group'))
    # ipdb.set_trace()
    if support_task_num != support_task_num:
        logging.warn('Tasks may be unbalanced!')
        if abs(support_task_num - query_task_num) >= 3:
            raise "Extremely unbalanced support & query split, Terminated!"
    #  Merge to list
    for task_id in range(0, min(support_task_num, query_task_num)):
        s_label = 'support.' + domain + '_' + str(task_id)
        q_label = 'query.' + domain + '_' + str(task_id)
        task_df_list.append(group_supoort_df[group_supoort_df.task_group == s_label].append(group_query_df[group_query_df.task_group == q_label]).reset_index(drop=True))
    for idx,i in enumerate(task_df_list):
        task_df_score_list.append(i.score.mean())
    task_with_score = list(zip(task_df_list, task_df_score_list))
    # ipdb.set_trace()
    if cl_sample:
        task_with_score.sort(key=lambda x : x[1])
    # TODO: PAUSE
    # ipdb.set_trace()
    return task_with_score

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


def read_tasks_ordered_by_curriculum(domains, source_dictionary, target_dictionary, args,
                                     meta_train_subset, meta_dev_subset, meta_test_subset, root, downstream_train_subset, downstream_valid_subset, downstream_test_subset, cl_learn):
    tasks_by_domain = OrderedDict()
    dev_by_domain = OrderedDict()
    # rng = np.random.RandomState(seed)
    meta_train_df_list = []
    for domain in domains:
        meta_train, meta_dev = load_static_meta_splits(root=root, domain=domain,
                                                       meta_train_subset=meta_train_subset,
                                                       meta_dev_subset=meta_dev_subset,
                                                       meta_test_subset=meta_test_subset,
                                                       downstream_train_subset=downstream_train_subset,
                                                       downstream_valid_subset=downstream_valid_subset,
                                                       downstream_test_subset=downstream_test_subset)
        meta_train_df_list.append(meta_train)
        # Split into task (store in Dataframe) list
        meta_dev.dev.task_group = downstream_valid_subset + '.' + domain
        # meta_train_with_score = split_domain_task(meta_train, support_tokens=args.support_tokens, query_tokens=args.query_tokens, domain=domain, cl_sample=cl_learn)
        # meta_train = [i[0] for i in meta_train_with_score]
        # meta_train_score = [i[1] for i in meta_train_with_score]
        meta_dev = split_domain_task(meta_dev, support_tokens=args.support_tokens, query_tokens=args.query_tokens, domain=domain, cl_sample=cl_learn, is_dev=True)
        # Convert data frames to fairseq tasks
        convert_to_fairseq_task_fn = partial(to_fairseq_task, args=args, src_dict=source_dictionary, tgt_dict=target_dictionary)
        # meta_train = list(map(convert_to_fairseq_task_fn, meta_train, meta_train_score))
        # # ipdb.set_trace()
        meta_dev = list(map(convert_to_fairseq_task_fn, meta_dev, [0 for i in range(0, len(meta_dev))]))
        # tasks_by_domain[domain] = meta_train
        dev_by_domain[domain] = meta_dev
        # print('[{}] Number of meta-training tasks: {} '.format(domain, len(meta_train)))
        print('[{}] Number of dev sentences: {} '.format(domain, len(meta_dev[0].user_data_frame)))

    # ======= Merge meta-train for spliting
    cl_merge_df = merge_meta_train(meta_train_df_list)
    meta_train_with_score = split_currciulum_task(cl_merge_df, support_tokens=args.support_tokens, query_tokens=args.query_tokens, cl_sample=cl_learn)
    meta_train = [i[0] for i in meta_train_with_score]
    meta_train_score = [i[1] for i in meta_train_with_score]
    convert_to_fairseq_task_fn = partial(to_fairseq_task, args=args, src_dict=source_dictionary, tgt_dict=target_dictionary)
    meta_train = list(map(convert_to_fairseq_task_fn, meta_train, meta_train_score))
    tasks_by_domain['CL'] = meta_train
    print('Total number of tasks: ', sum([len(i) for i in tasks_by_domain.values()]))
    print('Total number of dev tasks: ', sum([len(i) for i in dev_by_domain.values()]))
    # ipdb.set_trace()
    return tasks_by_domain, dev_by_domain

def read_tasks_ordered_by_domain(domains, source_dictionary, target_dictionary, args,
                                 meta_train_subset, meta_dev_subset, meta_test_subset, root, downstream_train_subset,downstream_valid_subset, downstream_test_subset, cl_learn):
    tasks_by_domain = OrderedDict()
    dev_by_domain = OrderedDict()
    # rng = np.random.RandomState(seed)
    for domain in domains:
        meta_train, meta_dev = load_static_meta_splits(root=root, domain=domain,
                                                       meta_train_subset=meta_train_subset,
                                                       meta_dev_subset=meta_dev_subset,
                                                       meta_test_subset=meta_test_subset,
                                                       downstream_train_subset=downstream_train_subset,
                                                       downstream_valid_subset=downstream_valid_subset,
                                                       downstream_test_subset=downstream_test_subset)
        # Split into task (store in Dataframe) list
        meta_dev.dev.task_group = downstream_valid_subset + '.' + domain
        meta_train_with_score = split_domain_task(meta_train, support_tokens=args.support_tokens, query_tokens=args.query_tokens, domain=domain, cl_sample=cl_learn)
        meta_train = [i[0] for i in meta_train_with_score]
        meta_train_score = [i[1] for i in meta_train_with_score]
        meta_dev = split_domain_task(meta_dev, support_tokens=args.support_tokens, query_tokens=args.query_tokens, domain=domain, cl_sample=cl_learn, is_dev=True)
        # Convert data frames to fairseq tasks
        convert_to_fairseq_task_fn = partial(to_fairseq_task, args=args, src_dict=source_dictionary, tgt_dict=target_dictionary)
        meta_train = list(map(convert_to_fairseq_task_fn, meta_train, meta_train_score))
        # ipdb.set_trace()
        meta_dev = list(map(convert_to_fairseq_task_fn, meta_dev, [0 for i in range(0, len(meta_dev))]))
        # meta_test = list(islice(map(convert_to_fairseq_task_fn, meta_test), test_limit))
        tasks_by_domain[domain] = meta_train
        dev_by_domain[domain] = meta_dev
        print('[{}] Number of meta-training tasks: {} '.format(domain, len(meta_train)))
        print('[{}] Number of dev sentences: {} '.format(domain, len(meta_dev[0].user_data_frame)))
        # print('Length of meta-dev: ', len(meta_dev))
    print('Total number of tasks: ', sum([len(i) for i in tasks_by_domain.values()]))
    print('Total number of dev tasks: ', sum([len(i) for i in dev_by_domain.values()]))
    # ipdb.set_trace()
    return tasks_by_domain, dev_by_domain


def prepare_meta_tasks(meta_learning_args, downstream_args):
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
    if meta_learning_args.is_curriculum and meta_learning_args.split_by_cl:
        tasks_by_domain, dev_by_domain = read_tasks_ordered_by_curriculum(domains=domains,
                                                                          source_dictionary=source_dictionary,
                                                                          target_dictionary=target_dictionary,
                                                                          args=downstream_args,
                                                                          meta_train_subset=meta_learning_args.train_subset,
                                                                          meta_dev_subset=meta_learning_args.valid_subset,
                                                                          meta_test_subset=meta_learning_args.test_subset,
                                                                          root=root,
                                                                          downstream_train_subset=downstream_args.train_subset,
                                                                          downstream_valid_subset=downstream_args.valid_subset,
                                                                          downstream_test_subset=downstream_args.test_subset,
                                                                          cl_learn=meta_learning_args.is_curriculum)
        cl_domains = ['CL']
        all_meta_train = apply_roundrobin(cl_domains, tasks_by_domain, cl_task_reorder=meta_learning_args.is_curriculum)
    else:
        tasks_by_domain, dev_by_domain = read_tasks_ordered_by_domain(domains=domains,
                                                                      source_dictionary=source_dictionary,
                                                                      target_dictionary=target_dictionary,
                                                                      args=downstream_args,
                                                                      meta_train_subset=meta_learning_args.train_subset,
                                                                      meta_dev_subset=meta_learning_args.valid_subset,
                                                                      meta_test_subset=meta_learning_args.test_subset,
                                                                      root=root,
                                                                      downstream_train_subset=downstream_args.train_subset,
                                                                      downstream_valid_subset=downstream_args.valid_subset,
                                                                      downstream_test_subset=downstream_args.test_subset,
                                                                      cl_learn=meta_learning_args.is_curriculum)
        all_meta_train = apply_roundrobin(domains, tasks_by_domain, cl_task_reorder=meta_learning_args.is_curriculum)
    all_dev_task = []
    for domain in domains:
        all_dev_task += dev_by_domain[domain]
    return all_meta_train, all_dev_task

# def prepare_and_balance_tasks(meta_learning_args, downstream_args):
#     domains = meta_learning_args.domains
#     print('domains:')
#     print(domains)
#     assert len(meta_learning_args.data) == 1
#     root = meta_learning_args.data[0]
#     train_limit = meta_learning_args.train_limit
#     valid_limit = meta_learning_args.valid_limit
#     test_limit = meta_learning_args.test_limit
#     tasks_by_domain = prepare_meta_tasks(root=root,
#                                          args=downstream_args,
#                                          domains=domains,
#                                          meta_train_subset=meta_learning_args.train_subset,
#                                          meta_dev_subset=meta_learning_args.valid_subset,
#                                          meta_test_subset=meta_learning_args.test_subset,
#                                          downstream_train_subset=downstream_args.train_subset,
#                                          downstream_valid_subset=downstream_args.valid_subset,
#                                          downstream_test_subset=downstream_args.test_subset)
#     balanced_meta_train, balanced_meta_dev, balanced_meta_test = apply_roundrobin(domains,
#                                                                                   tasks_by_domain,
#                                                                                   train_limit=train_limit,
#                                                                                   valid_limit=valid_limit,
#                                                                                   test_limit=test_limit)
#     return balanced_meta_train, balanced_meta_dev, balanced_meta_test

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
    maybe_downstream_argv = argv[split_index+1:]
    parser = options.get_meta_training_parser(default_task='meta-curriculum')
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

def main_meta_learning(meta_learning_args, downstream_args, fine_tune_args):
    meta_tasks, meta_dev_tasks = prepare_meta_tasks(meta_learning_args=meta_learning_args, downstream_args=downstream_args)

    def load_meta_tasks_fn():
        return meta_tasks,meta_dev_tasks
        # return balanced_meta_train, balanced_meta_dev, balanced_meta_test

    meta_learning_utils.run_maybe_distributed_reptile(meta_learning_args=meta_learning_args,
                                                      downstream_args=downstream_args,
                                                      load_meta_tasks_fn=load_meta_tasks_fn,
                                                      fine_tune_args=fine_tune_args)

if __name__ == "__main__":
    cli_main(main_fn=main_meta_learning)
    # cli_main(main_fn=main_meta_adaptation)
