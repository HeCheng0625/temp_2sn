


export LANG=en_US.UTF-8
export LANGUAGE=
export LC_CTYPE="en_US.UTF-8"
export LC_NUMERIC=zh_CN.UTF-8
export LC_TIME=zh_CN.UTF-8
export LC_COLLATE="en_US.UTF-8"
export LC_MONETARY=zh_CN.UTF-8
export LC_MESSAGES="en_US.UTF-8"
export LC_PAPER=zh_CN.UTF-8
export LC_NAME=zh_CN.UTF-8
export LC_ADDRESS=zh_CN.UTF-8
export LC_TELEPHONE=zh_CN.UTF-8
export LC_MEASUREMENT=zh_CN.UTF-8
export LC_IDENTIFICATION=zh_CN.UTF-8
export LC_ALL=

sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y
sudo apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo locale-gen zh_CN.UTF-8

sudo apt-get install -y vim

# PHILLY_USER=${USER}
# echo "PHILLY_USER:${PHILLY_USER}"


# sudo rm /etc/sudoers.d/${PHILLY_USER}
# sudo touch /etc/sudoers.d/${PHILLY_USER}
# sudo chmod 777 /etc/sudoers.d/${PHILLY_USER}
# sudo echo "Defaults        secure_path=\"$path:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/${PHILLY_USER}
# sudo chmod 0440 /etc/sudoers.d/${PHILLY_USER}


source ~/.bashrc
conda create -n knn_mt python=3.8 -y

conda activate knn_mt
conda create -n knn_mt python=3.8 -y
conda activate knn_mt
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install scipy h5py -y
conda install matplotlib -y
pip install -r requirements.txt
 
conda install pytorch-lightning -c conda-forge -y


TEXT=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/raw_data/LJSpeech-1.1
DATA_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/data-bin/ljspeech_discrete_res16hop200_vqemb_44w
SAVE_DIR=/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/checkpoints/ljspeech_latent_onlyenc

# /bin/rm -rf ${SAVE_DIR}

mkdir -p ${SAVE_DIR}
export CUDA_VISIBLE_DEVICES=0
python -m usr_dir.tasks.latent_diffusion --config configs/tts/lj/fs2_latent.yaml --exp_name ${SAVE_DIR} --reset --hparams "data_dir=${DATA_DIR},raw_data_dir=${TEXT},max_tokens=30000,use_pitch_embed=False,use_energy_embed=False"

#TEXT=${EXP_HOME}/raw_data/ljspeech_res2
#DATA_DIR=${EXP_HOME}/data-bin/ljspeech_ps_res2