description: fs2 bin train split

# target:
#   service: amlk8s
#   name: itplabrr1cl1
#   vc: resrchvc

target:
  service: sing
  name: msrresrchvc
  workspace_name: msrresrchws

environment:
  registry: docker.io # docker.io
  image: npuichigo/pytorch:pytorch1.8.1-py38-cuda11.1
  setup:
    # sleeping for 30 seconds to improve Philly stability
    - sleep 30
    # install rsync for log sync
    - sudo apt-get update
    - sudo apt-get install rsync -y
    # install mpi4py for distributed training
    - pip install --user mpi4py
    # enabling IB in case it needs to be used
    - export NCCL_IB_DISABLE=0
    # maybe install the current package in develop mode
    - pip install --user --no-deps -e .
    # sometimes wait for other nodes to be ready
    - sleep 10
    # we can add any other additional set up here (e.g. exawatt preparation)
storage:
  blob:
    storage_account_name: msramldl
    container_name:  ml-dl
    mount_dir: /blob_nouse

search:
  job_template:
    name: exp_{experiment_name:s}_{auto:3s}
    sku: 8C32
    priority: high
    submit_args:
      constraints:
        - tag: connectivityDomain
          type: uniqueConstraint
      container_args:
        shm_size_per_gpu: 5000000000
        shm_size: 256G
      env:
        MKL_THREADING_LAYER: GNU
        SHARED_MEMORY_PERCENT: 0.5
    command:
      - sudo apt-get install git -y
      - sudo chmod 777 -R /opt/conda
      - cd /home/aiscuser # cd /home/aiscuser
      - pwd
      - sudo apt-get install blobfuse
      - cp -r /blob_nouse/v-shenkai/envs/fuse_connection.cfg .
      - sudo mkdir /blob
      - sudo mkdir /mnt/teamdrive_tmp
      - sudo chown aiscuser:aiscuser /mnt/teamdrive_tmp
      - sudo chown aiscuser:aiscuser /blob
      - blobfuse /blob --tmp-path=/mnt/teamdrive_tmp --config-file=/home/aiscuser/fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
      - cd /home/aiscuser
      - cp /blob/v-zeqianju/.bashrc .
      - source /opt/conda/etc/profile.d/conda.sh
      - bash /blob/v-zeqianju/git_init.sh
      - git clone git@github.com:AlanSwift/fast_speech2.git
      - cd fast_speech2/
      - git fetch
      - git checkout zq
      - git pull origin zq:zq
      - ls shell_scripts/latent
      - conda create -n knn_mt python=3.8 -y
      - conda activate knn_mt
      - conda install scipy h5py -y
      - conda install matplotlib -y
      - pip install -r requirements.txt
      - conda install pytorch-lightning -c conda-forge -y
      - pip install pysptk 
      - pip install setuptools==59.5.0
      - pip install scikit-image
      - pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
      - pip install tensorboard -U
      - conda install -c anaconda libstdcxx-ng
      - sudo cp /opt/conda/envs/knn_mt/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 
      - bash shell_scripts/latent/preprocess_hop200_product_full_train.sh {split}      
  type: grid
  max_trials: 1000
  params:
     - name: split
       values: ['0', '1', '2', '3', '4', '5', '6', '7']
