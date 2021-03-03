CUDA=$1
domains=("covid" "bible" "books" "ecb" "ted")
epoch=20                                                                        
save_dir=/home/zhanrunzhe/data/outputs/meta_cl_by_cl                          
out_dir=/home/zhanrunzhe/data/domain_meta_bin                                   
for domain in ${domains[@]}; do
    echo eval domain $domain
    echo "--------------Score for domain" $domain "----------------"
    CUDA_VISIBLE_DEVICES=$CUDA python ~/workspace/meta-mt/generate.py  $out_dir/$domain \
        --source-lang en --target-lang de \
        --path $save_dir/checkpoint_best.pt --score-reference \
        --max-sentences 128 --beam 5 --remove-bpe sentencepiece --gen-subset train \
        | grep ^H | sed 's/^H\-//' | sort -n -k 1  > /home/zhanrunzhe/data/domain_meta_data/$domain/meta_split/meta-test-spm/unseen.score # | cut -f 2
     echo "-------------- Done domain" $domain "----------------"             
 done
