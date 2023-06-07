


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/data-bin/ljspeech_discrete_res16hop200_vqemb_44w_fixcode
SAVE_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/codebookloss_nodetach_dmu_0_dwn_0_dmelw_1_dnoisew_1_vqweight_0.1_vq_dist_weight_0_dila_1_pe_scale_1/

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0
python -m usr_dir.tasks.latent_diffusion --config configs/tts/lj/fs2_latent.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=30000,pe_scale=1" --infer
