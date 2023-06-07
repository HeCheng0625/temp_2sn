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

save_prefix=product_5w_conditionLNfilm_newref_spkdropout_wavnetprenormsa
reference_encoder_filters="32 32 64 64 128 128 256 256 512 512"
arch='8 8 8 8 8 8 8 8 8 8 8 8'
enc_layers=6
dec_layers=6
residual_channels=512
residual_layers=40
hidden_size=512

TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/home/xuta/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fixcode_debug_newf0/
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
use_new_refenc=True
use_pitch_embed=False
predictor_hidden=${hidden_size}
spk_dropout=0.2
diffusion_has_sattn=True
SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/${save_prefix}_refenc_dmu_${detach_mu}_dwn_${detach_wavenet}_dmelw_${diffusion_mel_weight}_dnoisew_${diff_loss_noise_weight}_vqweight_${vq_quantizer_weight}_vq_dist_weight_${vq_dist_weight}_dila_${dilation_cycle_length}_pe_scale_${pe_scale}
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_WORKERS=5
python -m usr_dir.tasks.latent_diffusion --config configs/tts/product/latent_diffusion_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=18000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},use_ref_enc=True,use_spk_embed=True,enc_layers=${enc_layers},dec_layers=${dec_layers},arch=${arch},residual_channels=${residual_channels},residual_layers=${residual_layers},hidden_size=${hidden_size},max_updates=100000000,\
                                            use_amp=True,accumulate_grad_batches=6,log_interval=100,prior_use_cln=True,diffusion_use_film=True,use_pitch_embed=${use_pitch_embed},predictor_type=sawavnet,diffusion_has_sattn=${diffusion_has_sattn},reference_encoder_filters=${reference_encoder_filters},remove_bos=True,use_new_refenc=${use_new_refenc},predictor_hidden=${predictor_hidden},spk_dropout=${spk_dropout},diffusion_sa_per_layer=2" 2>&1 | tee -a ${SAVE_DIR}/train.log

#TEXT=${EXP_HOME}/raw_data/ljspeech_res2

