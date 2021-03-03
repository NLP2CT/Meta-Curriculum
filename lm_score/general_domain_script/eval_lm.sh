fairseq-eval-lm ~/data/data-bin/wmt19_lm \
    --path ~/data/lm_outputs/general_lm/checkpoint_best.pt \
    --batch-size 20 \
    --tokens-per-sample 512 \
    --context-window 400
