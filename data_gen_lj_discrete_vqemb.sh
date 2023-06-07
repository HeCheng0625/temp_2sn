

TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/data-bin/ljspeech_discrete_res16hop200_vqemb_44w
VQ_CKPT=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/infered_lj_440000/rvq_hop200.pt
mkdir -p ${DATA_DIR}


python -m datasets.tts.lj.gen_fs2_discrete_vqemb_hop200res16 --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},vq_ckpt=${VQ_CKPT}" --config configs/tts/lj/fs2_discrete_vqemb_hop200res16.yaml

#TEXT=${EXP_HOME}/raw_data/ljspeech_res2
#DATA_DIR=${EXP_HOME}/data-bin/ljspeech_ps_res2
