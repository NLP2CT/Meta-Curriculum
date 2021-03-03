#!/bin/bash
# domains=("bible" "wmt-news" "ted" "subtitles" "kde" "jrc" "globalvoices" "emea" "ecb" "books")
domains=("emea" "globalvoices" "jrc" "kde" "wmt-news")

SCRIPTS=/home/zhanrunzhe/tools/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=/home/zhanrunzhe/tools/subword-nmt/subword_nmt
BPE_TOKENS=40000
lang=en

BPE_CODE=/home/zhanrunzhe/data/g_domain/prep/code
data_all=/home/zhanrunzhe/data/domain_lm_en


for domain in ${domains[@]}; do
	data_dir=$data_all/$domain/
    rm -rf $data_dir/bpe/
    mkdir -p $data_dir/bpe/
    for f in train.clean valid.clean test.clean; do
        echo "apply_bpe.py to ${domain} ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $data_dir/$f.$lang > $data_dir/bpe/bpe.$f.$lang
    done
done
