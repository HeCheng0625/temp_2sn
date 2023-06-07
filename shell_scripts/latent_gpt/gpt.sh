





save_prefix="5w_ar"


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1

# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fastbin
DATA_DIR=~/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s

SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_gpt/${save_prefix}
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_WORKERS=3
# configs/tts/product/latent_gpt_5wdata.yaml
python -m usr_dir.tasks.uni_speech_gpt --config configs/tts/product/latent_gpt_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=4000,log_interval=100,\
            use_amp=True,accumulate_grad_batches=24,\
            remove_bos=True,remove_eos=False,append_sep=True,online_prepocess=True" 2>&1 | tee ${SAVE_DIR}/train.log



