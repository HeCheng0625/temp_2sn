FULL_FILE=/blob/v-zeqianju/dataset/tts/product_5w/10000_h.txt

CODE_DATA=/blob/v-zeqianju/dataset/tts/product_5w/10000h_code
EMB_DATA=/blob/v-zeqianju/dataset/tts/product_5w/10000h_emb

TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
<<<<<<< HEAD:shell_scripts/latent/preprocess_hop200_product_10000.sh
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_10000_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta
DATA_DIR=/home/v-zeqianju/data-bin/product_10000_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta
=======
DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_10000_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta
# DATA_DIR=/home/v-zeqianju/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta
>>>>>>> update binarization:shell_scripts/latent/preprocess_hop200_product_full_wotrain.sh
VQ_CKPT=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/infered_lj_440000/rvq_hop200.pt
mkdir -p ${DATA_DIR}

export CUDA_VISIBLE_DEVICES=0
python -m datasets.tts.product.gen_fs2_product_full_wotrain --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},vq_ckpt=${VQ_CKPT},full_file=${FULL_FILE},code_data_dir=${CODE_DATA},emb_data_dir=${EMB_DATA},f0_use_product=False" --config configs/tts/product/fs2_discrete_vqemb_hop200res16.yaml

#TEXT=${EXP_HOME}/raw_data/ljspeech_res2
#DATA_DIR=${EXP_HOME}/data-bin/ljspeech_ps_res2

# cp phone_set.json ${DATA_DIR}/
