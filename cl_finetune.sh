export CUDA_VISIBLE_DEVICES=2

code_dir=/home/zhanrunzhe/workspace/meta-mt
DARA_DIR=/home/zhanrunzhe/data/domain_meta_data
GPUS=1
CRITERION=fomaml
BASELINE= #empty
META_DEV="--valid-subset meta-dev-spm"
# DOMAINS="emea globalvoices" bible covid books ecb ted emea globalvoices jrc
# DOMAINS=("bible")
DOMAINS=("ted" "emea" "globalvoices" "jrc" "kde" "wmt-news")
PRE_TRAIN=/home/zhanrunzhe/data/outputs/mixed-epoch20/checkpoint_avg.pt

outspace=/home/zhanrunzhe/data/outputs/meta_cl_by_cl8k
finetune_out=/home/zhanrunzhe/data/outputs/meta_cl_ad_8k
META_TRAIN_PT=$outspace/checkpoint_best.pt
PT_LOGS_DIR=logs/
PT_OUTPUT_DIR=$outspace
PT_DATA_DIR=
MODEL=

OPTIMIZER=adam
META_LR=0.00001
DO_SAVE= #empty
FT_EPOCH=20
SAVEINTERVALUPDATES=1000

SACREBLEU="--sacrebleu"
MAX_UPDATE= #empty
LRSCHEDULER=fixed_reduce_lr_on_plateau
BEAM=5

FP16=
FP16_INIT=
SCALE=
TIME=$(date "+%Y-%m-%d-%H:%M:%S")
DEBUG= #-m ipdb

# mkdir -p $outspace
# mkdir -p $outspace/log
rm -rf $PT_LOGS_DIR/tensorboard_mc_cl8k
mkdir -p $PT_LOGS_DIR/tensorboard_mc_cl8k
# max-tokens = max-task per batch

for domain in ${DOMAINS[@]}; do
    echo finetuning $domain
    sleep 3
    python -u $DEBUG $code_dir/cl_finetune.py $DARA_DIR\
        --test-subset meta-test-spm --train-subset meta-train-spm $META_DEV --arch transformer \
        --criterion $CRITERION $BASELINE \
        --domains $domain --max-tokens 1 \
        --is-curriculum --split-by-cl --distributed-world-size $GPUS \
        --required-batch-size-multiple 1 \
        --optimizer $OPTIMIZER --lr $META_LR $DO_SAVE \
        --save-dir $PT_OUTPUT_DIR --save-interval-updates $SAVEINTERVALUPDATES \
        --max-epoch 1 \
        --skip-invalid-size-inputs-valid-test \
        --flush-secs 1 --train-percentage 0.99 --log-format json \
        --- --restore-file $META_TRAIN_PT --task user_translation --arch transformer --is-curriculum --train-subset support --test-subset query --valid-subset dev_sub --max-tokens 2048 --update-freq 8 --skip-invalid-size-inputs-valid-test --distributed-world-size 1 --max-epoch $FT_EPOCH $MAX_UPDATE --save-interval 5 --remove-bpe sentencepiece --optimizer adam --lr 5e-05 --lr-scheduler $LRSCHEDULER --support-tokens 8000 --query-tokens 16000 --source-lang en --target-lang de --beam 5 --quiet --log-format json --save-dir $finetune_out  | tee $PT_LOGS_DIR/cl_finetune-$TIME.log
    sleep 3
done

