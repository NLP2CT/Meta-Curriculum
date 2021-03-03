import torch
import os
import gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from fairseq.models.transformer_lm import TransformerLanguageModel

def load_lm(save_path, checkpoint_name, bpe_code):
    lm = TransformerLanguageModel.from_pretrained(save_path, checkpoint_name , tokenizer='moses', bpe='fastbpe', bpe_codes=bpe_code)
    lm.eval()
    lm.cuda()
    return lm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Adaptation distance scoring.')
    parser.add_argument('--general-lm', help='general NLM path', required=True)
    parser.add_argument('--domain-lms', help='domain NLMs path', required=True)
    parser.add_argument('--bpe-code', help='bpe code for training the lm', required=True)
    parser.add_argument('--data-path', help='domain data path', required=True)
    parser.add_argument('--domains', nargs="+", default=["jrc"], help='domain data path',required=True)
    args = parser.parse_args()

    domains = args.domains
    print("loading general lm...")
    general_model_file_path = args.general_lm
    g_lm = load_lm(general_model_file_path, 'checkpoint_best.pt', args.bpe_code)
    for domain in domains:
        invalid_cnt = 0
        g_score_list = []
        d_score_list = []
        print("loading domain{0} lm...".format(domain))
        domain_model_file_path = os.path.join(args.domain_lms, domain)
        d_lm = load_lm(domain_model_file_path, 'checkpoint_best.pt', args.bpe_code)
        file_name = os.path.join(args.data_path, domain, 'split','clean_tok',domain + '.de-en.en')
        print(file_name)
        f = open(file_name,'r')
        src_sentences = f.readlines()
        for sentence in tqdm(src_sentences):
            try:
                general_sc = g_lm.score(sentence.strip())['positional_scores'].mean().neg().exp().item()
                domain_sc = d_lm.score(sentence.strip())['positional_scores'].mean().neg().exp().item()
                g_score_list.append(general_sc)
                d_score_list.append(domain_sc)
            except Exception:
                print("Exception Occur")
                invalid_cnt += 1
                g_score_list.append(-99999.0)
                d_score_list.append(-99999.0)
        assert len(g_score_list) == len(d_score_list)
        g_score_list = np.array(g_score_list)
        d_score_list = np.array(d_score_list)
        f = open(os.path.join(args.data_path, domain, 'split','clean_tok','g_score.pkl'), 'wb')
        pickle.dump(g_score_list, f)
        f.close()
        f = open(os.path.join(args.data_path, domain, 'split','clean_tok','d_score.pkl'), 'wb')
        pickle.dump(d_score_list, f)
        f.close()
        diff_socre = g_score_list - d_score_list
        score_info_df = pd.DataFrame({'general_domain_score':g_score_list, 'domain_score':d_score_list, 'diff_score':diff_socre})
        score_info_df.to_csv(os.path.join(args.data_path, domain,'split','clean_tok','grading_info.csv'))
        del d_lm
        gc.collect()
        torch.cuda.empty_cache()
        print("Invaild sample: ", invalid_cnt)