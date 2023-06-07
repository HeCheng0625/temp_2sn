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

save_prefix=mls_librivox_v1_fromscratch
# save_prefix=debug
reference_encoder_filters="64 128 256 512"
arch='8 8 8 8 8 8'
enc_layers=6
dec_layers=0
residual_channels=512
residual_layers=40
hidden_size=512

TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
# DATA_DIR=/home/xuta/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fixcode_debug_newf0/
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
DATA_DIR=/blob/v-yuancwang/TTS_Data/11lab_process
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/mls_libri_v2
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_1000_discrete_res16hop200_vqemb_44w_fixcode_fixbug

use_new_refenc=True
use_pitch_embed=True
predictor_hidden=${hidden_size}
spk_dropout=0.0
ref_random_clip=False
dur_cln=True
pitch_cln=True
duration_layers=30
pitch_layers=30
diffusion_from_prior=False
prior_weight=0.0
predictor_use_res=True
use_ref_enc=False
ref_enc_arch="8 8 8 8 8 8"  # decoder arch for ref enc
prior_use_cln=False
diffusion_ca_per_layer=3
predictor_use_cattention=True
predictor_ca_per_layer=3
use_spk_prompt=True

ref_query_tokens=32
diff_attn_type=cln
ref_query_norm=True

noise_factor=3

SAVE_DIR=/blob/v-yuancwang/LATENT_TTS/finetune_11lab_v3
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_WORKERS=10
export NCCL_NSOCKS_PERTHREAD=20
export NCCL_SOCKET_NTHREADS=10
# configs/tts/product/latent_diffusion_5wdata.yaml

# codec_ckpt=/blob/v-yuancwang/checkpoint-2324000.pt,\
python -m usr_dir.tasks.latent_diffusion_pl2 --config configs/tts/product/latent_diffusion_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},\
                                            raw_data_dir=${TEXT},max_tokens=14000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},\
                                            diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},use_spk_embed=True,enc_layers=${enc_layers},dec_layers=${dec_layers},arch=${arch},residual_channels=${residual_channels},residual_layers=${residual_layers},hidden_size=${hidden_size},max_updates=100000000,\
                                            use_amp=True,log_interval=100,prior_use_cln=${prior_use_cln},diffusion_use_film=True,use_pitch_embed=${use_pitch_embed},diffusion_has_sattn=False,reference_encoder_filters=${reference_encoder_filters},remove_bos=True,use_new_refenc=${use_new_refenc},predictor_hidden=${predictor_hidden},spk_dropout=${spk_dropout},ref_random_clip=${ref_random_clip},dur_cln=${dur_cln},pitch_cln=${pitch_cln},duration_layers=${duration_layers},pitch_layers=${pitch_layers},diffusion_from_prior=${diffusion_from_prior},prior_weight=${prior_weight},predictor_use_res=${predictor_use_res},\
                                            strategy=ddp,accumulate_grad_batches=1,num_nodes=1,warmup_updates=30000,\
                                            lr=5e-5,max_frames=3000,max_input_tokens=600,use_random_segment_as_ref=True,\
                                            vq_ckpt=/blob/v-yuancwang/new_rvq_generator/rvq_hop200.pt,\
                                            vocoder_ckpt=/blob/v-yuancwang/new_rvq_generator/generator_hop200.pt,\
                                            noise_factor=${noise_factor},load_opt=False,\
                                            use_ref_enc=${use_ref_enc},ref_enc_arch=${ref_enc_arch},skip_decoder=True,query_attn_type=independent_w_mha,\
                                            predictor_type=wavnet_crossattn,diff_attn_type=${diff_attn_type},diffusion_ca_per_layer=${diffusion_ca_per_layer},\
                                            ref_query_norm=${ref_query_norm},ref_query_tokens=${ref_query_tokens},predictor_use_cattention=${predictor_use_cattention},predictor_ca_per_layer=${predictor_ca_per_layer},use_spk_prompt=${use_spk_prompt}" 2>&1 | tee -a ${SAVE_DIR}/train.log

# transformer_esitimator_arch='13 13 13 13 13 13 13 13 13 13 13 13'
# predictor_type=transformer_refcat_conformer,transformer_esitimator_arch=${transformer_esitimator_arch},dec_ffn_kernel_size=3,transformer_hidden=512