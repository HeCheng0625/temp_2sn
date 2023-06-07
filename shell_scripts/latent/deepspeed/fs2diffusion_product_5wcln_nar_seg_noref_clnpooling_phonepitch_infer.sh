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

save_prefix=product_5w_v1.1big_segnoref_per4_new-clnpooling_phonepitch
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
DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=~/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_1000_discrete_res16hop200_vqemb_44w_fixcode_fixbug

use_new_refenc=True
use_pitch_embed=True
predictor_hidden=${hidden_size}
spk_dropout=0.0
ref_random_clip=False
dur_cln=False
pitch_cln=False
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
pitch_predictor_type=conv_phone

diff_attn_type=cln_pooling
SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/${save_prefix}_refenc_dmu_${detach_mu}_dwn_${detach_wavenet}_dmelw_${diffusion_mel_weight}_dnoisew_${diff_loss_noise_weight}_vqweight_${vq_quantizer_weight}_vq_dist_weight_${vq_dist_weight}_dila_${dilation_cycle_length}_pe_scale_${pe_scale}
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
export NUM_WORKERS=1 #5
export NCCL_NSOCKS_PERTHREAD=20
export NCCL_SOCKET_NTHREADS=10
# configs/tts/product/latent_diffusion_5wdata.yaml

temp=1.2
test_set_name=valid  # non-zero-shot equal to zero dur to prpeprocess bugs, use valid as seen spk setting
# test_set_name=non-zero_shot_test
infer_laoban=7
in_context_infer=False
stoc=False
ref_norm=False
inference_ckpt=0-82000.ckpt

python -m usr_dir.tasks.latent_diffusion_pl2 --config configs/tts/product/latent_diffusion_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=8000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},use_spk_embed=True,enc_layers=${enc_layers},dec_layers=${dec_layers},arch=${arch},residual_channels=${residual_channels},residual_layers=${residual_layers},hidden_size=${hidden_size},max_updates=100000000,\
                                            use_amp=True,accumulate_grad_batches=20,log_interval=100,prior_use_cln=${prior_use_cln},diffusion_use_film=True,use_pitch_embed=${use_pitch_embed},predictor_type=wavnet_crossattn,diffusion_has_sattn=False,reference_encoder_filters=${reference_encoder_filters},remove_bos=True,use_new_refenc=${use_new_refenc},predictor_hidden=${predictor_hidden},spk_dropout=${spk_dropout},ref_random_clip=${ref_random_clip},dur_cln=${dur_cln},pitch_cln=${pitch_cln},duration_layers=${duration_layers},pitch_layers=${pitch_layers},diffusion_from_prior=${diffusion_from_prior},prior_weight=${prior_weight},predictor_use_res=${predictor_use_res},\
                                            strategy=ddp,accumulate_grad_batches=1,num_nodes=1,\
                                            lr=5e-4,max_frames=3000,max_input_tokens=600,use_random_segment_as_ref=True,\
                                            use_ref_enc=${use_ref_enc},ref_enc_arch=${ref_enc_arch},skip_decoder=True,pitch_predictor_type=${pitch_predictor_type},\
                                            diff_attn_type=${diff_attn_type},diffusion_ca_per_layer=${diffusion_ca_per_layer},predictor_use_cattention=${predictor_use_cattention},predictor_ca_per_layer=${predictor_ca_per_layer},use_spk_prompt=${use_spk_prompt},\
                                            inference_ckpt=${inference_ckpt},train_set_name=valid,test_set_name=${test_set_name},infer_laoban=${infer_laoban},in_context_infer=${in_context_infer},stoc=${stoc},temp=${temp},ref_norm=${ref_norm}" --infer


