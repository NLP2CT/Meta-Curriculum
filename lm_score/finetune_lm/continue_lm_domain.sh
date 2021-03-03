set -e
# domains=("bible" "wmt-news" "ted" "subtitles" "kde" "jrc" "globalvoices" "emea" "ecb" "books")
# seen "emea" "jrc" "kde" "wmt-news" "globalvoices"s
domains=("wmt-news")
sl=en
tl=de
CUDA=0,1,2,3
data_all=/home/zhanrunzhe/data/domain_lm_en
bin_data=/home/zhanrunzhe/data/data-bin
out_dir=/home/zhanrunzhe/data/lm_outputs
fairseq_dir=/home/zhanrunzhe/workspace/diffculty-eval-nmt/fairseq/fairseq_cli

for domain in ${domains[@]}; do
        data_dir=$data_all/$domain/bpe
        bin_dir=$bin_data/$domain
        # rm -rf bin_dir
	# echo "--------------Binrary domain:" $domain "----------------"
	# CUDA_VISIBLE_DEVICES=$CUDA python $fairseq_dir/preprocess.py \
        # --only-source \
	# 	--trainpref $data_dir/bpe.train.clean.en \
	# 	--validpref $data_dir/bpe.valid.clean.en \
	# 	--testpref  $data_dir/bpe.test.clean.en \
	# 	--srcdict ~/data/data-bin/wmt19_lm/dict.txt \
	# 	--destdir $bin_dir \
	# 	--workers 20
        echo "-------------- Finetune LM in domain:" $domain "----------------"
        mkdir -p $out_dir/${domain}/log/
        CUDA_VISIBLE_DEVICES=$CUDA python $fairseq_dir/train.py --task language_modeling \
                $bin_dir \
                --restore-file $out_dir/general_lm4k/checkpoint_best.pt \
                --save-dir $out_dir/${domain} \
                --arch transformer_lm --share-decoder-input-output-embed \
                --dropout 0.1 \
                --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
                --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
                --tokens-per-sample 512 --sample-break-mode none \
                --max-tokens 4096 --update-freq 8 --max-update 3000 \
                --reset-dataloader --reset-optimizer --reset-lr-scheduler --reset-meters \
                --log-format json 2>&1 | tee $out_dir/${domain}/lm_$domain.log
        echo "-------------- Done LM in domain:" $domain ", wait for release gpu mem----------------"
        sleep 10
done
