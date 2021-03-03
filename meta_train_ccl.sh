export CUDA_VISIBLE_DEVICES=2

code_dir=/home/zhanrunzhe/workspace/meta-mt
DARA_DIR=/home/zhanrunzhe/data/domain_meta_data
GPUS=1
CRITERION=fomaml
BASELINE= #empty
META_DEV="--valid-subset meta-dev-spm"
# DOMAINS="emea globalvoices"
DOMAINS="emea globalvoices jrc kde wmt-news"
PRE_TRAIN=/home/zhanrunzhe/data/outputs/mixed-epoch20/checkpoint_avg.pt

outspace=/home/zhanrunzhe/data/outputs/meta_cl_by_cl8k-finalex
PT_LOGS_DIR=logs/
PT_OUTPUT_DIR=$outspace
PT_DATA_DIR=
MODEL=

OPTIMIZER=adam
META_LR=0.00001
DO_SAVE= #empty
SAVEINTERVALUPDATES=1000

SACREBLEU="--sacrebleu"
MAX_UPDATE= #empty
LRSCHEDULER=fixed_reduce_lr_on_plateau
BEAM=5

FP16=
FP16_INIT=
SCALE=
TIME=$(date "+%Y-%m-%d-%H:%M:%S")

# mkdir -p $outspace
# mkdir -p $outspace/log
TF_BOARD=$PT_LOGS_DIR/mc_cl8k_logex
rm -rf $TF_BOARD
mkdir -p $TF_BOARD
# max-tokens = max-task per batch

python -u $code_dir/meta_curriculum_train.py $DARA_DIR\
    --train-subset meta-train-spm $META_DEV --arch transformer \
    --criterion $CRITERION $BASELINE \
    --domains $DOMAINS --max-tokens 1 \
    --is-curriculum --split-by-cl --distributed-world-size $GPUS \
    --required-batch-size-multiple 1 \
    --tensorboard-logdir $TF_BOARD \
    --optimizer $OPTIMIZER --lr $META_LR $DO_SAVE \
    --save-dir $PT_OUTPUT_DIR --save-interval-updates $SAVEINTERVALUPDATES \
    --max-epoch 20 \
    --skip-invalid-size-inputs-valid-test \
    --flush-secs 1 --train-percentage 0.99 --restore-file $PRE_TRAIN --log-format json \
    --- --task user_translation --is-curriculum --train-subset support --test-subset query --valid-subset dev_sub --max-tokens 2000 --update-freq 10000 --skip-invalid-size-inputs-valid-test --distributed-world-size 1 --max-epoch 1 $MAX_UPDATE --optimizer adam --lr 5e-05 --lr-scheduler $LRSCHEDULER --no-save --support-tokens 8000 --query-tokens 16000 --source-lang en --target-lang de | tee $PT_LOGS_DIR/LOG_FRAC_mc_cl-$TIME.log
