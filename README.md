# Compact Bidirectional Transformer for Image Captioning

## Requirements
- Python 3.8
- Pytorch 1.6
- lmdb
- h5py
- tensorboardX

## Prepare Data
1. Please use **git clone --recurse-submodules** to clone this repository and remember to follow initialization steps in coco-caption/README.md.
2. Download the preprocessd dataset from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ESjYq2E7NlJGuyCaNyCSadEBarCtcRtUMR7Nd0UgTIm3-A?e=Rl0Bu2) and extract it to data/.
3. Please download the converted [VinVL](https://github.com/pzzhang/VinVL/blob/main/DOWNLOAD.md#pre-exacted-image-features) feature from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ETEt4voFu-hAkxxwbwgZEz0BrGiDizFmqMTf3wGdWlv0bQ?e=FcqLhg) and place them under data/mscoco_VinVL/. You can also optionally follow this [instruction](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md#convert-from-peteanderson80s-original-file) to prepare the fixed or adaptive  bottom-up features extracted by Anderson and place them under data/mscoco/ or data/mscoco_adaptive/.
4. Download part checkpoints from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ER1w9q3ekqpKmiVPW_yL2pABY2TSyb_PoyBK0xDqEHH_zg?e=7Hfwdk) and extract them to save/.

## Offline Evaluation
To reproduce the results of a model, such as 'cit-pair-decoder-1-data-aug', just run

```
python  eval.py  --model  save/cit-pair-decoder-1-data-aug/model-best.pth   --infos_path  save/cit-pair-decoder-1-data-aug/infos_cit-pair-decoder-1-data-aug-best.pkl      --beam_size   3   --id  cit-pair-decoder-1-data-aug   --split val
```


## Training
1.  In the first training stage, such as training 'cit-pair-decoder-1-data-aug' model ,  just run 
```
python  train.py   --noamopt --noamopt_warmup 20000   --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0  --scheduled_sampling_start 0  --save_checkpoint_every 3000 --language_eval 1  --val_images_use -1  --max_epochs 15     --checkpoint_path   save/cit-pair-decoder-1-data-aug  --id   cit-pair-decoder-1-data-aug    --mode  pair    --caption_model   cit    --lang_inter_weight  0   --lang_inter_af   relu
```

2. Then in the second training stage, please copy the above pretrained model first

```
cd save
./copy_model.sh  cit-pair-decoder-1-data-aug    nsc-cit-pair-decoder-1-data-aug
cd ..
``` 
and then run
```
python  train.py    --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 14  --max_epochs    30  --start_from   save/nsc-cit-pair-decoder-1-data-aug     --checkpoint_path   save/nsc-cit-pair-decoder-1-data-aug   --id  nsc-cit-pair-decoder-1-data-aug   --caption_model  cit   --mode  pair   --lang_inter_weight  0   --lang_inter_af   relu
```

## Note
1. You can use the `git reflog` to list all commits and use `git reset --hard  commit_id` to change to corresponding commit. 

## Citation

```

```

## Acknowledgements
This repository is built upon [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). Thanks for the released  code.
