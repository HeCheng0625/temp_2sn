


gpt_speech_dropout=0.5


save_prefix="5w_ar_dropout5"


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1

# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fastbin
DATA_DIR=~/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s

SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_gpt/${save_prefix}_spdrop${gpt_speech_dropout}
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_WORKERS=5
python -m usr_dir.tasks.uni_speech_gpt --config configs/tts/product/latent_gpt_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=3000,log_interval=100,\
            use_amp=True,accumulate_grad_batches=30,\
            gpt_speech_dropout=${gpt_speech_dropout},\
            remove_bos=True,remove_eos=False,append_sep=True,online_prepocess=True" 2>&1 | tee ${SAVE_DIR}/train.log



