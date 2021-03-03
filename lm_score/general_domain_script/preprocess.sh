#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh


SCRIPTS=/home/zhanrunzhe/tools/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=/home/zhanrunzhe/tools/subword-nmt/subword_nmt
BPE_TOKENS=40000
lang=en

mkdir -p tmp prep

echo "pre-processing data..."
for l in train valid test; do
    rm tmp/$l.$lang.tok
    cat $l.$lang | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 16 -a -l $lang >> tmp/$l.$lang.tok
done

TRAIN=tmp/train.en.tok
BPE_CODE=prep/code

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $lang; do
    for f in train.$lang.tok valid.$lang.tok test.$lang.tok; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < tmp/$f > bpe.$f
    done
done

# perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
# perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

# for L in $src $tgt; do
#     cp $tmp/bpe.test.$L $prep/test.$L
# done
