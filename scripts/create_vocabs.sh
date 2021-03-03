#!/usr/bin/env bash
# cat all training data together
#cat auto_ecom_med_tech_ted_wmt.train.en auto_ecom_med_tech_ted_wmt.train.de | shuf > train.all
#wc -l *train.*

#train spm model
#echo training spm model
#spm_train --bos_id=-1 --eos_id=0 --unk_id=1 --vocab_size 40000 --input train.all --model_prefix joint.40k

#spm_encode --model=joint.40k.model --output_format piece auto_ecom_med_tech_ted_wmt.train.en > train.spm.en
#spm_encode --model=joint.40k.model --output_format piece auto_ecom_med_tech_ted_wmt.train.de > train.spm.de

#spm_encode --model=joint.40k.model --output_format piece auto_ecom_med_tech_ted_wmt.dev.en > dev.spm.en
#spm_encode --model=joint.40k.model --output_format piece auto_ecom_med_tech_ted_wmt.dev.de > dev.spm.de

DEST=$DATA/2019-06-22-144200/


echo creating data for fairseq
python /tmp/pycharm_project_989/preprocess.py --trainpref $DATA/train.spm --validpref $DATA/dev.spm --testpref $DATA/test.spm -s en -t de --workers 48 --joined-dictionary --thresholdtgt 4 --thresholdsrc 4 --destdir $DEST
echo done creating fairseq data