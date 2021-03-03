set -e
# domains=("bible" "wmt-news" "ted" "subtitles" "kde" "jrc" "globalvoices" "emea" "ecb" "books")
# seen "emea" "jrc" "kde" "wmt-news" "globalvoices"s
# domains=("emea" "jrc" "kde" "wmt-news" "globalvoices")
domains=('general_lm4k')
sl=en
tl=de
CUDA=3
data_all=/home/zhanrunzhe/data/domain_lm_en
bin_data=/home/zhanrunzhe/data/data-bin
out_dir=/home/zhanrunzhe/data/lm_outputs
fairseq_dir=/home/zhanrunzhe/workspace/diffculty-eval-nmt/fairseq/fairseq_cli

for domain in ${domains[@]}; do
        data_dir=$data_all/$domain/bpe
        bin_dir=$bin_data/wmt19_lm
	echo "-------------- Eval LM in domain:" $domain "----------------"
	CUDA_VISIBLE_DEVICES=$CUDA python $fairseq_dir/eval_lm.py $bin_dir \
		--gen-subset test --path $out_dir/$domain/checkpoint_best.pt --sample-break-mode eos \
		--max-tokens 2048 --context-window 128
	sleep 5
	echo $CUDA_VISIBLE_DEVICES
	CUDA_VISIBLE_DEVICES=$CUDA python $fairseq_dir/eval_lm.py $bin_dir \
                --gen-subset valid --path $out_dir/$domain/checkpoint_best.pt --sample-break-mode eos \
                --max-tokens 2048 --context-window 128
	echo "-------------- Done LM in domain:" $domain ", wait for release gpu mem----------------"
        sleep 10
done
