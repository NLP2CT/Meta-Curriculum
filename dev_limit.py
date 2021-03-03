import os
import random

DATA_DIR = '/home/zhanrunzhe/data/domain_meta_data_16k_32'
SEED = 7

def sents_reader(file):
    f = open(file, encoding='utf8')
    lines = f.readlines()
    f.close()
    return [i.strip() for i in lines]

def read_para_dev_data(domain):
    path_prefix = os.path.join(DATA_DIR, domain, 'meta_split', 'meta-dev-spm', 'dev')
    source_lang = sents_reader(path_prefix + '.en')
    target_lang = sents_reader(path_prefix + '.de')
    para_data = list(zip(source_lang, target_lang))
    return para_data

def write_sents(para_list, meta_split_dir, set_type, source_lang='en', target_lang='de'):
    src_writer = open(os.path.join(meta_split_dir, set_type + '.' + source_lang), 'w', encoding='utf8')
    tgt_writer = open(os.path.join(meta_split_dir, set_type + '.' + target_lang), 'w', encoding='utf8')
    score_writer = open(os.path.join(meta_split_dir, set_type + '.' + 'score'), 'w', encoding='utf8')
    for idx, para_tuple in enumerate(para_list):
        src_sent, tgt_sent = para_tuple
        src_writer.write(src_sent + '\n')
        tgt_writer.write(tgt_sent + '\n')

if __name__ == "__main__":
    random.seed(SEED)
    domains = ['covid','bible','books','ecb','ted', 'emea','globalvoices','jrc','kde','wmt-news']
    for domain in domains:
        domain_para = read_para_dev_data(domain)
        random.shuffle(domain_para)
        domain_max_len = len(domain_para)
        if domain_max_len > 5000:
            domain_max_len = 5000
        print("Domain: {}, Dev_LEN:{}".format(domain, domain_max_len))
        down_sample = domain_para[:domain_max_len]
        TO_PATH = os.path.join(DATA_DIR, domain, 'meta_split', 'meta-dev-spm')
        write_sents(down_sample, TO_PATH, 'dev_sub')
