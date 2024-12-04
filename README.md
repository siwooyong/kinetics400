# kinetics400
simple pytorch pipeline for pretraining/finetuning vision models on kinetics-400


## data
download the kinetics-400 dataset from [opendatalab dataset](https://opendatalab.com/OpenMMLab/Kinetics-400) and arrange the data as follows \
for more details, check `data/`

```
data/
├── Kinetics-400.tar.gz
├── preprocess.py
└── README.md
```

## setup
```
apt install libgl1-mesa-glx
pip install -r requirements.txt
```

## preprocess
unzip the kinetics-400 located in the `data/`
```
python -m data.preprocess
```

after preprocessing, the directory structure will be as follows
```
data/
├── Kinetics-400.tar.gz
├── preprocess.py
├── README.md
└── Kinetics-400
    ├── videos_train/
    ├── videos_val/
    ├── kinetics400_train_list_videos.txt
    └── kinetics400_val_list_videos.txt
```

## supervised learning
classic supervised learning on kinetics-400 dataset
```
python -m train --save_dir weights \
                --model_name MCG-NJU/videomae-base-finetuned-kinetics \
                --pretrained \
                --pretrained_name MCG-NJU/videomae-base \
                --pretrained_dir pretrained_weights \
                --n_epoch 100 \
                --batch_size 32 \
                --lr 3e-4 \
                --n_worker 16 \
                --n_device 8 \
                --precision bf16-mixed \
                --dtype bfloat16 \
                --strategy ddp \
                --save_frequency 5 \
                --label_smoothing 0.1 \
                --input_size 224 
```

## self-supervised learning
currently, only [videomae](https://arxiv.org/abs/2203.12602) is supported for self-supervised learning
```
python -m pretraining --save_dir pretrained_weights \
                      --model_name MCG-NJU/videomae-base \
                      --n_epoch 400 \
                      --batch_size 64 \
                      --lr 5e-4 \
                      --n_worker 16 \
                      --n_device 8 \
                      --precision bf16-mixed \
                      --dtype bfloat16 \
                      --strategy ddp \
                      --save_frequency 20 \
                      --input_size 224 \
                      --mask_ratio 0.9 \
                      --tubelet_size 2 \
                      --norm_pix_loss
```

## results
- original repo performs self-supervised training for 800 epochs, while this repo achieves similar performance in just 400 epochs
- check `results/` 

- `videomae_vit_base`
    - it takes about 60 hours for pretraining using `8 x RTX 4090`
    - it takes about 18 hours for finetuning using `8 x RTX 4090`

|metric|top1_acc|top5_acc|
|---|---|---|
|this repo|78.64|93.65|
|official repo|79.99|94.42|

![image](results/videomae_vit_base/finetuning/loss_curve.png)

## acknowledgement
this project makes use of the following libraries and models
- [timm](https://github.com/huggingface/pytorch-image-models)
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
- [transformers](https://github.com/huggingface/transformers)
- [jepa](https://github.com/facebookresearch/jepa)