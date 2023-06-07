


gpt_speech_dropout=0.5


save_prefix="deepspeed_5w_ar_dropout5"


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1

# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fastbin
# DATA_DIR=~/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s

SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_gpt/${save_prefix}_spdrop${gpt_speech_dropout}
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
export NCCL_NSOCKS_PERTHREAD=10
export NCCL_SOCKET_NTHREADS=5
export NUM_WORKERS=1

test_set_name=non-zero_shot_test
infer_laoban=True
in_context_infer=False
inference_ckpt="0-3000.ckpt"

python -m usr_dir.tasks.uni_speech_gpt_pl2 --config configs/tts/product/latent_gpt_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=8000,log_interval=100,\
            strategy=deepspeed,precision=32,accumulate_grad_batches=20,val_check_interval=20000,lr=0.0005,\
            gpt_speech_dropout=${gpt_speech_dropout},\
            remove_bos=True,remove_eos=False,append_sep=True,online_prepocess=True,test_set_name=${test_set_name},train_set_name=valid,inference_ckpt=${inference_ckpt},in_context_infer=${in_context_infer}" --infer



