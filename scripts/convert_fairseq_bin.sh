CLUSTER_ROOT=/home/zhanrunzhe/data/cluster
code_dir=/home/zhanrunzhe/code/meta-mt/
domains=("bible")
#domains=("bible" "wmt-news" "ted" "subtitles" "kde" "jrc" "globalvoices" "emea" "ecb" "books")

create_fairseq_data_dir() {
    if [ $# -eq 4 ]; then # If we have four parameters
        python $code_dir/preprocess.py --trainpref $1 --validpref $2 --testpref $3 -s en -t de --workers 48 --destdir $4 --srcdict $SRCDICT --tgtdict $TGTDICT
    else
        python $code_dir/preprocess.py --trainpref $1 --validpref $2 -s en -t de --workers 48 --destdir $3 --srcdict $SRCDICT --tgtdict $TGTDICT
    fi
}

createfairseqcluster=true
if $createfairseqcluster; then
    echo creating fairseq bin data for all domains
    for domain in $domains; do
        echo creating fairseq data for $domain ...
        create_fairseq_data_dir $CLUSTER_ROOT/${domain}_de-en/train.spm $CLUSTER_ROOT/${domain}_de-en/dev.spm $CLUSTER_ROOT/${domain}_de-en/fairseq
    done
fi