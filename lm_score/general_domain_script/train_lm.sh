mkdir /home/zhanrunzhe/workspace/lm_log
TRAIN_CLI=/home/zhanrunzhe/workspace/diffculty-eval-nmt/fairseq/fairseq_cli/train.py
python $TRAIN_CLI --task language_modeling \
  ~/data/data-bin/wmt19_lm \
  --save-dir ~/data/lm_outputs/general_lm4k\
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 4096 --update-freq 16 \
  --max-update 50000 2>&1 | tee /home/zhanrunzhe/workspace/lm_log/lm_general.log

