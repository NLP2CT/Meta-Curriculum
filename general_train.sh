export CUDA_VISIBLE_DEVICES=2,3
sl=en
tl=de
dataset='mixed'
data_dir=/home/zhanrunzhe/data/fairseq-bin/${dataset}/

epoch=20
save_dir=/home/zhanrunzhe/data/outputs/${dataset}-epoch${epoch}/
mkdir -p $save_dir
CUDA_VISIBLE_DEICES=2,3 python ~/code/meta-mt/train.py $data_dir \
              --save-dir $save_dir \
              --arch transformer \
              --source-lang ${sl} --target-lang ${tl} \
              --weight-decay 0.0001 \
              --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
              --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
              --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
              --lr 0.0007 --min-lr 1e-9 \
	      --max-update 100000 \
	      --max-tokens 2048 --update-freq 16 \
              --max-epoch ${epoch} --save-interval 1 \
              --log-format json  1> $save_dir/log 2> $save_dir/err

