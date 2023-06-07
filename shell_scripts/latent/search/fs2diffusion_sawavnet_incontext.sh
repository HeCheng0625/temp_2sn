


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

# sudo apt-get update -y
# sudo apt-get upgrade -y
# sudo apt-get install -y
# sudo apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
# sudo apt-get install -y locales
# sudo locale-gen en_US.UTF-8
# sudo locale-gen zh_CN.UTF-8

# sudo apt-get install -y vim

# # PHILLY_USER=${USER}
# # echo "PHILLY_USER:${PHILLY_USER}"


# # sudo rm /etc/sudoers.d/${PHILLY_USER}
# # sudo touch /etc/sudoers.d/${PHILLY_USER}
# # sudo chmod 777 /etc/sudoers.d/${PHILLY_USER}
# # sudo echo "Defaults        secure_path=\"$path:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/${PHILLY_USER}
# # sudo chmod 0440 /etc/sudoers.d/${PHILLY_USER}


# source ~/.bashrc
# conda create -n knn_mt python=3.8 -y

# conda activate knn_mt

# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
# conda install scipy h5py -y
# conda install matplotlib -y
# pip install -r requirements.txt
 
# conda install pytorch-lightning -c conda-forge -y
# pip install pysptk



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

skip_decoder=0

save_prefix=sawavnet
predictor_type=sawavnet


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/data-bin/ljspeech_discrete_res16hop200_vqemb_44w_fixcode




SAVE_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/kai_srfixbug/debug

# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0
python -m usr_dir.tasks.latent_diffusion --config configs/tts/lj/fs2_latent.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=16000,detach_mu=${detach_mu},detach_wavenet=${detach_wavenet},diffusion_mel_weight=${diffusion_mel_weight},diff_loss_noise_weight=${diff_loss_noise_weight},vq_quantizer_weight=${vq_quantizer_weight},vq_dist_weight=${vq_dist_weight},dilation_cycle_length=${dilation_cycle_length},pe_scale=${pe_scale},skip_decoder=${skip_decoder},predictor_type=${predictor_type},incontext=True"

#TEXT=${EXP_HOME}/raw_data/ljspeech_res2
#DATA_DIR=${EXP_HOME}/data-bin/ljspeech_ps_res2
