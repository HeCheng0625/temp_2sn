# parameter search

## detach part
detach_mu=0
detach_wavenet=0

## diffusion loss
diffusion_mel_weight=1
diff_loss_noise_weight=1

## codebook loss
vq_quantizer_weight=0.1
vq_dist_weight=0

## dilation
dilation_cycle_length=2
pe_scale=1

save_prefix=product_100


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_100_discrete_res16hop200_vqemb_44w_fixcode
# DATA_DIR=/home/v-shenka/product_24_discrete_res16hop200_vqemb_44w_fixcode



SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/${save_prefix}_refenc_dmu_${detach_mu}_dwn_${detach_wavenet}_dmelw_${diffusion_mel_weight}_dnoisew_${diff_loss_noise_weight}_vqweight_${vq_quantizer_weight}_vq_dist_weight_${vq_dist_weight}_dila_${dilation_cycle_length}_pe_scale_${pe_scale}

# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_WORKERS=10
<<<<<<< HEAD
<<<<<<< HEAD
python -m usr_dir.tasks.latent_diffusion --config configs/tts/lj/fs2_latent.yaml --exp_name ${SAVE_DIR} --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=16000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},max_tokens=30000,use_ref_enc=True,use_spk_embed=True"
=======
python -m usr_dir.tasks.latent_diffusion --config configs/tts/lj/fs2_latent.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=16000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},max_tokens=30000,use_ref_enc=True,use_spk_embed=True"
>>>>>>> fix ref spk embed
=======
python -m usr_dir.tasks.latent_diffusion --config configs/tts/lj/fs2_latent.yaml --exp_name ${SAVE_DIR} --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=16000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},max_tokens=30000,use_ref_enc=True,use_spk_embed=True"
>>>>>>> update

