fairseq-preprocess --only-source \
    --trainpref bpe.train.en.tok --validpref bpe.valid.en.tok --testpref bpe.test.en.tok \
    --destdir ~/data/data-bin/wmt19_lm \
    --workers 20