cd

cp /blob/v-shenkai/utils/azcopy .

./azcopy copy "https://msramldl.blob.core.windows.net/ml-dl/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s?sv=2021-08-06&st=2023-03-22T17%3A34%3A54Z&se=2024-03-23T17%3A34%3A00Z&sr=c&sp=rl&sig=y8EewYYfUtSWkmGmt7PzzHctPTCXmHQxlapDyDrQ0vc%3D" . --recursive





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

git config --global user.name jzq2000
git config --global user.email juzeqian@mail.ustc.edu.cn

cp /blob/v-zeqianju/id_rsa $HOME
sudo chmod 700 $HOME/id_rsa
cp $HOME/.ssh/config $HOME/.ssh/config_old
cat /blob/v-zeqianju/ssh.config $HOME/.ssh/config_old > $HOME/.ssh/config

cp -r /blob/v-shenkai/envs/.tmux* .

source ~/.bashrc
conda create -n deepspeed python=3.8 -y

conda activate deepspeed
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install scipy h5py -y
conda install matplotlib -y
conda install pytorch-lightning==2.0 -c conda-forge -y
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

