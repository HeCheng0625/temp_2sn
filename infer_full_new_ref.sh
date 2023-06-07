


# export LANG=en_US.UTF-8
# export LANGUAGE=
# export LC_CTYPE="en_US.UTF-8"
# export LC_NUMERIC=zh_CN.UTF-8
# export LC_TIME=zh_CN.UTF-8
# export LC_COLLATE="en_US.UTF-8"
# export LC_MONETARY=zh_CN.UTF-8
# export LC_MESSAGES="en_US.UTF-8"
# export LC_PAPER=zh_CN.UTF-8
# export LC_NAME=zh_CN.UTF-8
# export LC_ADDRESS=zh_CN.UTF-8
# export LC_TELEPHONE=zh_CN.UTF-8
# export LC_MEASUREMENT=zh_CN.UTF-8
# export LC_IDENTIFICATION=zh_CN.UTF-8
# export LC_ALL=

# #sudo apt-get update -y
# #sudo apt-get upgrade -y
# #sudo apt-get install -y
# #sudo apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
# #sudo apt-get install -y locales
# #sudo locale-gen en_US.UTF-8
# #sudo locale-gen zh_CN.UTF-8

# #sudo apt-get install -y vim

# PHILLY_USER=${USER}
# echo "PHILLY_USER:${PHILLY_USER}"


# #sudo rm /etc/sudoers.d/${PHILLY_USER}
# #sudo touch /etc/sudoers.d/${PHILLY_USER}
# #sudo chmod 777 /etc/sudoers.d/${PHILLY_USER}
# #sudo echo "Defaults        secure_path=\"$path:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/${PHILLY_USER}
# #sudo chmod 0440 /etc/sudoers.d/${PHILLY_USER}


# source ~/.bashrc
# conda create -n knn_mt python=3.8 -y

# conda activate knn_mt
# # conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
# conda install scipy h5py -y
# conda install matplotlib -y
# pip install -r requirements.txt

# conda install pytorch-lightning -c conda-forge -y
# pip install pysptk
# pip install setuptools==59.5.0
# pip install scikit-image
# pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# pip install tensorboard -U
# pip install librosa
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

save_prefix=product_5w_conditionLNfilm_newref_debug
reference_encoder_filters="32 32 64 64 128 128 256 256 512 512"
arch='8 8 8 8 8 8 8 8 8 8 8 8'
enc_layers=6
dec_layers=6
residual_channels=512
residual_layers=40
hidden_size=512

TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
# DATA_DIR=/home/xuta/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
# DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_24_discrete_res16hop200_vqemb_44w_fixcode_debug_newf0/
DATA_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s
use_new_refenc=True
use_pitch_embed=False
predictor_hidden=${hidden_size}
spk_dropout=0.001  # train =0.0, infer >0.0  due to compatitity

SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/${save_prefix}_refenc_dmu_${detach_mu}_dwn_${detach_wavenet}_dmelw_${diffusion_mel_weight}_dnoisew_${diff_loss_noise_weight}_vqweight_${vq_quantizer_weight}_vq_dist_weight_${vq_dist_weight}_dila_${dilation_cycle_length}_pe_scale_${pe_scale}
# SAVE_DIR=/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug
# /bin/rm -rf ${SAVE_DIR}
# test_set_name=valid  # non-zero-shot equal to zero dur to prpeprocess bugs, use valid as seen spk setting
# test_set_name=non-zero_shot_test
infer_laoban=False
in_context_infer=False
# mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0
export NUM_WORKERS=1
dur_cln=True
pitch_cln=True

python -m usr_dir.tasks.latent_diffusion --config configs/tts/product/latent_diffusion_5wdata.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=23000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},use_ref_enc=True,use_spk_embed=True,enc_layers=${enc_layers},dec_layers=${dec_layers},arch=${arch},residual_channels=${residual_channels},residual_layers=${residual_layers},hidden_size=${hidden_size},max_updates=100000000,\
                                            use_amp=False,accumulate_grad_batches=6,log_interval=100,prior_use_cln=True,diffusion_use_film=True,use_pitch_embed=${use_pitch_embed},predictor_type=sawavnet,diffusion_has_sattn=False,reference_encoder_filters=${reference_encoder_filters},remove_bos=True,use_new_refenc=${use_new_refenc},predictor_hidden=${predictor_hidden},spk_dropout=${spk_dropout},train_set_name=valid,test_set_name=${test_set_name},infer_laoban=${infer_laoban},in_context_infer=${in_context_infer},dur_cln=${dur_cln},pitch_cln=${pitch_cln}" --infer 

#TEXT=${EXP_HOME}/raw_data/ljspeech_res2
#DATA_DIR=${EXP_HOME}/data-bin/ljspeech_ps_res2


