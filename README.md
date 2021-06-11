# Meta-Curriculum
Meta-Curriculum Learning for Domain Adaptation in Neural Machine Translation (AAAI 2021)

### Citation

Please cite as:

```bibtex
@inproceedings{zhan2021metacl,
  title={Meta-Curriculum Learning for Domain Adaptation in Neural Machine Translation},
  author={Zhan, Runzhe and Liu, Xuebo and Wong, Derek F. and Chao, Lidia S.},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={16},
  month={May}, 
  year={2021},
  pages={14310-14318}
}

```

### Requirements and Installation
This implementation is based on [fairseq(v0.6.2)](https://github.com/pytorch/fairseq/tree/v0.6.2/fairseq) and partial code from Sharaf, Hassan, and Daume III (2020).

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* CUDA & cudatoolkit >= 9.0

```
git clone https://github.com/NLP2CT/Meta-Curriculum
cd Meta-Curriculum
pip install --editable .
```

### Pipeline 
1. Train a baseline model following the SOP in `examples/translation/README.md`. See our script `general_train.sh` (also utilize it for baseline finetuning).
2. Use the scripts containing in the folder `lm_score/general_domain_script` to train a general domain NLM.
3. Finetune the domain-specific NLM following the script `lm_score/finetune_lm/continue_lm_domain.sh`.
4. Score the adaptation divergence for domain corpus:
```shell
CUDA_VISIBLE_DEVICES=0 python lm_score/finetune_lm/score.py --general-lm GENERAL DOMAIN NLM PATH --domain-lms DOMAIN NLMs PATH --bpe-code BPE CODE --data-path DOMAIN CORPUS PATH --domains [DOMAIN1, DOMAIN2, ...]
```
> Please note that you may separately run the LM training/scoring with higher version fairseq (>=0.9.0) due to the API changes.

5. Prepare meta-learning data set using `meta_data_prep.py`.
```shell
python meta_data_prep.py --data-path DOMAIN_DATA_PATH --split-dir META_SPLIT_SAVE_DIR
                              --spm-model SPM_MODEL_PATH --k-support N --k-query N
                              --meta_train_task N --meta_test_task N
                              --unseen-domains [UNSEEN_DOMAINS ...] 
                              --seen-domains [SEEN_DOMAINS ...]
```
6. (Meta-Train) Train meta-learning model with curriculum using the script `meta_train_ccl.sh`.
7. Score the unseen domains using the script `score_unseens.sh`.
8. (Meta-Test) Finetune meta-trained model with curriculum using the script `cl_finetune.sh`.


### 🌟 COVID-19 English-German Small-Scale Parallel Corpus
See `covid19-ende/covid_de.txt` and `covid19-ende/covid_en.txt` (Raw data without preprocessing).
