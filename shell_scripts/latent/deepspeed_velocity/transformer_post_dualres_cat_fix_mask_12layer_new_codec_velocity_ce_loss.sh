## default train script

detach_mu=0
detach_wavenet=0

## diffusion loss
# diffusion_mel_weight=1
diffusion_mel_weight=0
diff_loss_noise_weight=0

## codebook loss
# vq_quantizer_weight=0.1
vq_quantizer_weight=0.1
vq_dist_weight=0

## dilation
dilation_cycle_length=2
pe_scale=1

save_prefix=product_5w_v1.1big_segnoref_query32_small-cln_refnorm_sepw
# save_prefix=debug
reference_encoder_filters="64 128 256 512"
arch='8 8 8 8 8 8'
enc_layers=6
dec_layers=0
residual_channels=512
residual_layers=40
hidden_size=512

TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1   # not used
# DATA_DIR=/home/xuta/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fixcode_debug_newf0/
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=~/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_1000_discrete_res16hop200_vqemb_44w_fixcode_fixbug
# DATA_DIR=/blob/v-yuancwang/TTS_Data/mls_wmeta
# DATA_DIR=/blob/v-yuancwang/TTS_Data/11lab_process
DATA_DIR=~/mls_wmeta
# DATA_DIR=/blob/v-yuancwang/ns2/data-bin/11lab_process

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

# directly cat reference latent
use_spk_prompt=False
ref_left_pad=False

ref_query_tokens=32
diff_attn_type=cln
ref_query_norm=True

noise_factor=3

# for finetune
load_opt=True

# velocity
diffusion_type=velocity
diff_velocity_weight=1

# for transformer diffuser
transformer_esitimator_arch='13 13 13 13 13 13 13 13 13 13 13 13'
transformer_hidden=512
diff_transformer_num_head=8

# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/${save_prefix}_refenc_dmu_${detach_mu}_dwn_${detach_wavenet}_dmelw_${diffusion_mel_weight}_dnoisew_${diff_loss_noise_weight}_vqweight_${vq_quantizer_weight}_vq_dist_weight_${vq_dist_weight}_dila_${dilation_cycle_length}_pe_scale_${pe_scale}_ref_query_tokens${ref_query_tokens}
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# SAVE_DIR=/blob/v-yuancwang/LATENT_TTS/transformer_pre_fix_mask_12layer_new_codec_velocity
SAVE_DIR=/blob/v-yuancwang/ns2/checkpoints/transformers/transformer_post_dualres_cat_fix_mask_new_codec_velocity_ce_loss
# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_WORKERS=10
export NCCL_NSOCKS_PERTHREAD=20
export NCCL_SOCKET_NTHREADS=10
# configs/tts/product/latent_diffusion_5wdata.yaml

#/opt/conda/envs/control/bin/python
python -m usr_dir.tasks.latent_diffusion_pl2 --config configs/tts/product/latent_diffusion_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},\
                                            raw_data_dir=${TEXT},max_tokens=4800,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},\
                                            diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},\
                                            vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},\
                                            pe_scale=${pe_scale},use_spk_embed=True,enc_layers=${enc_layers},dec_layers=${dec_layers},arch=${arch},\
                                            residual_channels=${residual_channels},residual_layers=${residual_layers},hidden_size=${hidden_size},max_updates=100000000,\
                                            use_amp=True,log_interval=100,prior_use_cln=${prior_use_cln},diffusion_use_film=True,use_pitch_embed=${use_pitch_embed},\
                                            diffusion_has_sattn=False,reference_encoder_filters=${reference_encoder_filters},remove_bos=True,use_new_refenc=${use_new_refenc},\
                                            predictor_hidden=${predictor_hidden},spk_dropout=${spk_dropout},ref_random_clip=${ref_random_clip},dur_cln=${dur_cln},\
                                            pitch_cln=${pitch_cln},duration_layers=${duration_layers},pitch_layers=${pitch_layers},diffusion_from_prior=${diffusion_from_prior},\
                                            prior_weight=${prior_weight},predictor_use_res=${predictor_use_res},\
                                            strategy=ddp,precision=16-mixed,accumulate_grad_batches=4,num_nodes=1,warmup_updates=30000,\
                                            lr=4e-4,max_frames=3000,max_input_tokens=600,use_random_segment_as_ref=True,\
                                            noise_factor=${noise_factor},load_opt=${load_opt},\
                                            vq_ckpt=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/infered_lj_2324000/rvq_hop200.pt,\
                                            vocoder_ckpt=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/infered_lj_2324000/generator_hop200.pt,\
                                            predictor_type=transformer_post_dualres_cat_fix_mask,diff_velocity_weight=${diff_velocity_weight},\
                                            ref_left_pad=${ref_left_pad},diffusion_type=${diffusion_type},diff_transformer_num_head=${diff_transformer_num_head},\
                                            transformer_esitimator_arch=${transformer_esitimator_arch}, dec_ffn_kernel_size=3, transformer_hidden=${transformer_hidden},\
                                            use_ref_enc=${use_ref_enc},ref_enc_arch=${ref_enc_arch},skip_decoder=True,query_attn_type=independent_w_mha,\
                                            diff_attn_type=${diff_attn_type},diffusion_ca_per_layer=${diffusion_ca_per_layer},\
                                            ref_query_norm=${ref_query_norm},ref_query_tokens=${ref_query_tokens},predictor_use_cattention=${predictor_use_cattention},\
                                            predictor_ca_per_layer=${predictor_ca_per_layer},use_spk_prompt=${use_spk_prompt}" 2>&1 | tee -a ${SAVE_DIR}/train.log
                                    
# vq_ckpt=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/infered_lj_2324000/rvq_hop200.pt,\
# vocoder_ckpt=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/infered_lj_2324000/generator_hop200.pt,\
# DATA_DIR=v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/mls_wmeta               
# codec_ckpt=/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/checkpoint-2324000.pt