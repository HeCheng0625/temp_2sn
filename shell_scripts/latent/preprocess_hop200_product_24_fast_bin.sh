TRAIN_FILE=/blob/v-zeqianju/dataset/tts/product_5w/24_h.txt
VALID_FILE=/blob/v-zeqianju/dataset/tts/product_5w/valid.txt
TEST_FILE=/blob/v-zeqianju/dataset/tts/product_5w/test.txt
CODE_DATA=/blob/v-zeqianju/dataset/tts/product_5w/24h_code
EMB_DATA=/blob/v-zeqianju/dataset/tts/product_5w/24h_emb

TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fastbin
VQ_CKPT=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/infered_lj_440000/rvq_hop200.pt
mkdir -p ${DATA_DIR}


python -m datasets.tts.product.gen_fastbin --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},vq_ckpt=${VQ_CKPT},train_file=${TRAIN_FILE},valid_file=${VALID_FILE},test_file=${TEST_FILE},code_data_dir=${CODE_DATA},emb_data_dir=${EMB_DATA},f0_use_product=False" --config configs/tts/product/fs2_discrete_vqemb_hop200res16.yaml

#TEXT=${EXP_HOME}/raw_data/ljspeech_res2
#DATA_DIR=${EXP_HOME}/data-bin/ljspeech_ps_res2

