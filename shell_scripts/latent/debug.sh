


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/data-bin/ljspeech_discrete_res16hop200_vqemb_44w
SAVE_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/ljspeech_discrete_44wres16hop200_vqemb_debug3

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0
python -m usr_dir.tasks.latent_diffusion --config configs/tts/lj/fs2_latent.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=16000,detach_mu=False,detach_wavenet=True"
