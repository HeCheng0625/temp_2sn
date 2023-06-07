# README

# NOTE!!! VERY IMPORTANT !!!!

You need to **manually mount blob** (as shown in code/\*.yaml) to avoid OOM (due to blob cache)  if dataset is large (e.g. 5w h wav)

```bash
sudo apt-get install git -y
cd /home/aiscuser # cd /home/aiscuser
sudo apt-get install blobfuse
cp -r /blob_nouse/v-shenkai/envs/fuse_connection.cfg .
sudo mkdir /blob
sudo mkdir /mnt/teamdrive_tmp
sudo chown aiscuser:aiscuser /mnt/teamdrive_tmp
sudo chown aiscuser:aiscuser /blob
blobfuse /blob --tmp-path=/mnt/teamdrive_tmp --config-file=/home/aiscuser/fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
```

## Pipeline

wav -->***pretrained codec*** --> code --> ***fs2 binarizer***---> bin

AMLT_CODE_DIR= ``/blob/v-zeqianju/dataset/tts/product_5w/amlt_codes/``



### Step 1. Extract code from tars

codebase:  /blob/v-zeqianju/torchtts_noamlt/

1. modify  ``shell_codes/infer_product_data_full.sh``
   
   VARIABLE:   MODEL_DIR, MODRL_PATH

2. submit amlt yaml to sing    ``AMLT_CODE_DIR/codec.yaml``
   
   

### Step 2. Binarization

codebase:  /blob/v-zeqianju/fast_speech2

1. modify ``shell_scripts/latent/preprocess_hop200_product_full_wotrain.sh``   
   
   VARIABLE： CODE_DATA，DATA_DIR
   
   

2. modify ``shell_scripts/latent/preprocess_hop200_product_full_train.sh``   (parallel processing)

VARIABLE： CODE_DATA，DATA_DIR, BLOB_DIR



3. submit amlt yaml to sing  **generate meta data & dataset wo train** ``AMLT_CODE_DIR/bin_wotrain.yaml``
   
   5w h data  --> ~24h
   
   

4. submit amlt yaml to sing **generate train given meta (8 split)**``AMLT_CODE_DIR/bin_train_split.sh``
   
   5w h data --> 12h
   
   

5.  **merge split train** (suggest amlt -i ``AMLT_CODE_DIR/debug_manual_blob.yaml``)
   
    
   
   ```bash
   azcopy each split dir to $HOME/DATA_DIR_{split}
   
   cd ~/fast_speech2/
   
   python -m datasets.tts.product.merge_train $HOME/DATA_DIR train {END_SPLIT} {START_SPLIT}
   ```
   
   
   
   
