import os
import random
import pathlib
import logging
import datetime
import argparse
import sys
import simplejson as json
import pandas as pd
import sentencepiece as spm
import pickle
from collections import namedtuple

# logging.basicConfig(format='%(levelname)s: %(message)s',
#                     level=logging.DEBUG)

MetaData = namedtuple('MetaData',['support','query'])
SEED = 7

def init_logger(log_path, name="meta-data-prep"):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    logfile = os.path.join(log_path, "%s-%s.log" % (name, datetime.datetime.today()))
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setLevel(logging.INFO)
    root.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    root.addHandler(consoleHandler)
    logging.debug("Logging to %s" % logfile)

def sents_reader(file):
    f = open(file, encoding='utf8')
    lines = f.readlines()
    f.close()
    return [i.strip() for i in lines]

def read_para_data(domain, seen_status, data_dir):
    path_prefix = os.path.join(data_dir, domain, 'split', 'clean_tok', domain + '.de-en.')
    source_lang = sents_reader(path_prefix + 'en')
    target_lang = sents_reader(path_prefix + 'de')
    score_df = None
    if seen_status == 'seen':
        score_df = pd.read_csv(os.path.join(data_dir, domain, 'split', 'clean_tok', 'grading_info.csv'))
        assert len(score_df) == len(source_lang)
    assert len(source_lang) == len(target_lang)  # check data integrity
    para_data = list(zip(source_lang, target_lang))
    return para_data, score_df

def build_corpus_info(para_data, score_df):
    src_tokens = []
    tgt_tokens = []
    for idx, para_sent in enumerate(para_data):
        en_sent, de_sent = para_sent
        src_tokens.append(len(en_sent.split()))
        tgt_tokens.append(len(de_sent.split()))
    if score_df is not None:  # exclude unseen cases
        assert len(src_tokens) == len(score_df['diff_score'])
        diff_score = score_df['diff_score']
        domain_info_df = pd.DataFrame({
            'src_tokens': src_tokens,
            'tgt_tokens': tgt_tokens,
            'diff_score': diff_score
        }, index=list(range(0, len(para_data))))
    else:
        domain_info_df = pd.DataFrame({
            'src_tokens': src_tokens,
            'tgt_tokens': tgt_tokens,
        }, index=list(range(0, len(para_data))))
    return domain_info_df

def split_task_by_tokens(info_df, sample_tokens, used_index=[]):
    shuffle_df = info_df.sample(frac=1, random_state=SEED)
    cur_tokens = 0
    sampled_index = []
    for idx, row in shuffle_df.iterrows():
        if cur_tokens >= sample_tokens:
            break
        if idx not in used_index:
            sampled_index.append(idx)
            cur_tokens += row['src_tokens']
    # logging tokens
    logging.info('sampled_len: {}, sampled_toknes: {}, target_tokens: {}'.format(
        len(sampled_index), cur_tokens, sample_tokens
    ))
    return sampled_index

def split_dev_data(all_indices, used_indices):
    remain_indices = list(set(all_indices).difference(used_indices))
    logging.info('[Remaining Sentences] All: {} | Meta-Used: {} | Dev: {}'.format(
        len(all_indices),
        len(used_indices),
        len(remain_indices),
    ))
    return remain_indices

def dump_dict_to_file(name_emb, emb_filename):
    json.dump(name_emb, open(os.path.join(emb_filename, 'meta_split_info.json'), "w"))
    pickle.dump(name_emb, open(os.path.join(emb_filename, 'meta_split_info.pkl'), "wb"))

def write_sents(para_list, meta_split_dir, set_type, source_lang='en', target_lang='de', scores=None):
    src_writer = open(os.path.join(meta_split_dir, set_type + '.' + source_lang), 'w', encoding='utf8')
    tgt_writer = open(os.path.join(meta_split_dir, set_type + '.' + target_lang), 'w', encoding='utf8')
    score_writer = open(os.path.join(meta_split_dir, set_type + '.' + 'score'), 'w', encoding='utf8')
    if scores is not None:
        assert len(scores) == len(para_list)
    for idx, para_tuple in enumerate(para_list):
        src_sent, tgt_sent = para_tuple
        src_writer.write(src_sent + '\n')
        tgt_writer.write(tgt_sent + '\n')
        if scores is not None:
            score_writer.write(str(scores[idx]) + '\n')

def write_dev_data(para_data, domain_sample_info, split_dir, split_type, spm_model, is_spm=False):
    dev_sents = [para_data[i] for i in domain_sample_info['meta-dev']]
    meta_split_dir = os.path.join(split_dir, split_type)
    pathlib.Path(meta_split_dir).mkdir(parents=True, exist_ok=True)
    write_sents(dev_sents, meta_split_dir, 'dev', 'en', 'de')
    if is_spm:
        spm_dir = meta_split_dir + '-spm'
        pathlib.Path(spm_dir).mkdir(parents=True, exist_ok=True)
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_model)
        spm_dev = []
        for _, para_tuple in enumerate(dev_sents):
            en, de = para_tuple
            en_spm = ' '.join(sp.EncodeAsPieces(en))
            de_spm = ' '.join(sp.EncodeAsPieces(de))
            spm_dev.append((en_spm, de_spm))
        write_sents(spm_dev, spm_dir, 'dev', 'en', 'de')

def write_split_data(para_data, domain_sample_info, split_dir, split_type, spm_model, is_spm=False):
    if split_type not in ['meta-train', 'meta-test', 'meta-dev']:
        raise 'Not supported meta split type'
    meta_split_dir = os.path.join(split_dir, split_type)
    pathlib.Path(meta_split_dir).mkdir(parents=True, exist_ok=True)
    split_indices = domain_sample_info[split_type]
    support_sents = [para_data[i] for i in split_indices.support]
    query_sents = [para_data[i] for i in split_indices.query]
    if domain_sample_info['status'] == 'seen':
        scores = domain_sample_info[split_type + '-score']
        write_sents(support_sents, meta_split_dir, 'support', 'en', 'de', scores=scores.support)
        write_sents(query_sents, meta_split_dir, 'query', 'en', 'de', scores=scores.query)
    else:
        write_sents(support_sents, meta_split_dir, 'support', 'en', 'de')
        write_sents(query_sents, meta_split_dir, 'query', 'en', 'de')
    if is_spm:
        spm_dir = meta_split_dir + '-spm'
        pathlib.Path(spm_dir).mkdir(parents=True, exist_ok=True)
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_model)
        spm_support = []
        spm_query = []
        for _, para_tuple in enumerate(support_sents):
            en, de = para_tuple
            en_spm = ' '.join(sp.EncodeAsPieces(en))
            de_spm = ' '.join(sp.EncodeAsPieces(de))
            spm_support.append((en_spm, de_spm))
        for _, para_tuple in enumerate(query_sents):
            en, de = para_tuple
            en_spm = ' '.join(sp.EncodeAsPieces(en))
            de_spm = ' '.join(sp.EncodeAsPieces(de))
            spm_query.append((en_spm, de_spm))
        if domain_sample_info['status'] == 'seen':
            scores = domain_sample_info[split_type + '-score']
            write_sents(spm_support, spm_dir, 'support', 'en', 'de', scores=scores.support)
            write_sents(spm_query, spm_dir, 'query', 'en', 'de', scores=scores.query)
        else:
            write_sents(spm_support, spm_dir, 'support', 'en', 'de')
            write_sents(spm_query, spm_dir, 'query', 'en', 'de')

# Split the domain data into D_meta-train & D_meta-test
def meta_dataset_split(meta_train_task_N, meta_test_task_N, domains, support_tokens, query_tokens, split_dir, spm_model, data_dir):
    domain_type = []
    for key in domains.keys():
        domain_type += [(i,key) for i in domains[key]]
    for domain,seen_status in domain_type:
        used_data_indices = []
        logging.info("[Domain]:" + domain + "| [Type] " + seen_status + "| Start creating split...")
        split_dir = os.path.join(split_dir, domain, 'meta_split')
        domain_sample_info = {
            'status': seen_status,
            'support': str(support_tokens),
            'query': str(query_tokens),
            'meta-train-task': meta_train_task_N,
            'meta-train': None,
            'meta-train-score': None,
            'meta-test-task': meta_test_task_N,
            'meta-test': None,
            'meta-test-score': None,
            'meta-dev': None
        }
        para_data, score_df = read_para_data(domain, seen_status, data_dir)
        domain_info_df = build_corpus_info(para_data, score_df)
        # Must keep same for meta-test dataset
        logging.info("[Meta Test Support]")
        mtest_support = split_task_by_tokens(domain_info_df, meta_test_task_N * support_tokens)
        logging.info("[Meta Test Query]")
        used_data_indices += mtest_support
        mtest_query = split_task_by_tokens(
            info_df=domain_info_df,
            sample_tokens=meta_test_task_N * query_tokens,
            used_index=used_data_indices
        )
        used_data_indices += mtest_query
        domain_sample_info['meta-test'] = MetaData(support=mtest_support, query=mtest_query)
        # Rest ~ Meta-Train
        if seen_status == 'seen':  # Only seen domains need D_meta-train
            domain_sample_info['meta-test-score'] = MetaData(
                support=[domain_info_df.loc[i].diff_score for i in mtest_support],
                query=[domain_info_df.loc[i].diff_score for i in mtest_query]
            )
            logging.info("[Meta Train Support]")
            mtrain_support = split_task_by_tokens(
                info_df=domain_info_df,
                sample_tokens=meta_train_task_N * support_tokens,
                used_index=used_data_indices
            )
            used_data_indices += mtrain_support
            logging.info("[Meta Train Query]")
            mtrain_query = split_task_by_tokens(
                info_df=domain_info_df,
                sample_tokens=meta_train_task_N * query_tokens,
                used_index=used_data_indices
            )
            used_data_indices += mtrain_query
            domain_sample_info['meta-train'] = MetaData(support=mtrain_support, query=mtrain_query)
            domain_sample_info['meta-train-score'] = MetaData(
                support=[domain_info_df.loc[i].diff_score for i in mtrain_support],
                query=[domain_info_df.loc[i].diff_score for i in mtrain_query]
            )
            write_split_data(para_data, domain_sample_info, split_dir, 'meta-train', spm_model,is_spm=True)
        # Save sampling info
        logging.info("Check unused data for meta-dev")
        mdev = split_dev_data(all_indices=list(range(0, len(para_data))), used_indices=used_data_indices)
        domain_sample_info['meta-dev'] = mdev
        logging.info("Save domain info csv & dataset info json")
        pathlib.Path(split_dir).mkdir(parents=True, exist_ok=True)
        domain_info_df.to_csv(os.path.join(split_dir, 'domain_info_all.csv'))
        dump_dict_to_file(domain_sample_info, split_dir)
        # Write sampled tok file & SPM file
        logging.info("Write static files to {}".format(split_dir))
        write_split_data(para_data, domain_sample_info, split_dir, 'meta-test', spm_model, is_spm=True)
        write_dev_data(para_data, domain_sample_info, split_dir, 'meta-dev', spm_model, is_spm=True)
        logging.info("========Done Current Domain========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptation distance scoring.')
    parser.add_argument('--data-path', help='domain corpus', required=True)
    parser.add_argument('--split-dir', help='path to save meta data split', required=True)
    parser.add_argument('--spm-model', help='sentencepiece model path', required=True)
    parser.add_argument('--k-support', metavar='N', type=int, help='support set tokens (K)', required=True)
    parser.add_argument('--k-query', metavar='N', type=int, help='query set tokens (K)', required=True)
    parser.add_argument('--meta_train_task', metavar='N', type=int, help='number of meta-train tasks', required=True)
    parser.add_argument('--meta_test_task', metavar='N', type=int, help='number of meta-test tasks', required=True)
    parser.add_argument('--unseen-domains', nargs="+", default=["bible"], help='unseen domains',required=True)
    parser.add_argument('--seen-domains', nargs="+", default=["emea"], help='unseen domains',required=True)
    args = parser.parse_args()

    DATA_DIR = args.data_path
    SPLIT_DIR = args.split_dir
    SPM_MODEL_PATH = args.spm_model
    init_logger(SPLIT_DIR)

    domains = {
        'unseen': args.unseen_domains,
        'seen': args.seen_domains
    }
    k_support = args.k_support
    k_query = args.k_query
    meta_train_task_N = args.meta_train_task
    meta_test_task_N = args.meta_test_task

    logging.info("[Domains]: {}".format(domains['unseen'] + domains['seen']))
    logging.info("[Toknes]: support-{}k, query-{}k".format(k_support, k_query))
    meta_dataset_split(meta_train_task_N=meta_train_task_N,
                       meta_test_task_N=meta_test_task_N,
                       domains=domains,
                       support_tokens=k_support * 1000,
                       query_tokens=k_query * 1000,
                       split_dir=SPLIT_DIR,
                       spm_model=SPM_MODEL_PATH,
                       data_dir=DATA_DIR
                       )
