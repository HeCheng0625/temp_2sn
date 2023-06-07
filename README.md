# FastSpeech 2
Implementation of "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"

## Quick Start

1. Prepare dataset
    ```bash
    mkdir -p data/raw/
    cd data/raw/
    wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -zxf LJSpeech-1.1.tar.bz2
    cd ../../
    python datasets/tts/lj/prepare.py
    ```
2. Forced alignment
    ```bash
    # Download MFA first: https://montreal-forced-aligner.readthedocs.io/en/stable/aligning.html
    # unzip to montreal-forced-aligner
    ./montreal-forced-aligner/bin/mfa_train_and_align data/raw/LJSpeech-1.1/mfa_input data/raw/LJSpeech-1.1/dict_mfa.txt data/raw/LJSpeech-1.1/mfa_outputs -t ./montreal-forced-aligner/tmp -j 24
    ```

3. Build binary data

    ```bash
    # fs2
    PYTHONPATH=. python datasets/tts/lj/gen_fs2.py --config configs/tts/lj/fs2.yaml
    # fs2s
    PYTHONPATH=. python datasets/tts/lj/gen_fs2s.py --config configs/tts/lj/fs2s.yaml
    ```

5. Train FastSpeech 2 and 2s
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tasks/fs2.py --config configs/tts/lj/fs2.yaml --exp_name fs2_exp1 --reset
    
    CUDA_VISIBLE_DEVICES=0 python tasks/fs2s.py --config configs/tts/lj/fs2s.yaml --exp_name fs2s_exp1 --reset
    ```

6. Download pre-trained vocoder
    ```
    mkdir wavegan_pretrained
    ```
    download `checkpoint-1000000steps.pkl`, `config.yml`, `stats.h5` from https://drive.google.com/open?id=1XRn3s_wzPF2fdfGshLwuvNHrbgD0hqVS to `wavegan_pretrained/`
   
7. Inference
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tasks/fs2.py --config configs/tts/lj/fs2.yaml --exp_name fs2_exp1 --infer
    CUDA_VISIBLE_DEVICES=0 python tasks/fs2s.py --config configs/tts/lj/fs2s.yaml --exp_name fs2s_exp1 --infer
    ```