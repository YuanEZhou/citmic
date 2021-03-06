#! /bin/bash 
source /zhouyuanen/anaconda3/bin/activate  base
cd  /zhouyuanen/new/backup/export/citmic

#stage1
python  train.py   --noamopt --noamopt_warmup 20000   --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0  --scheduled_sampling_start 0  --save_checkpoint_every 3000 --language_eval 1  --val_images_use -1  --max_epochs 15     --checkpoint_path   save/cit-pair-decoder-1-data-aug  --id   cit-pair-decoder-1-data-aug    --mode  pair    --caption_model   cit    --lang_inter_weight  0   --lang_inter_af   relu

# # # stage2
# python  train.py    --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 14  --max_epochs    30  --start_from   save/nsc-cit-pair-decoder-1-check     --checkpoint_path   save/nsc-cit-pair-decoder-1-check   --id  nsc-cit-pair-decoder-1-check   --caption_model  cit   --mode  pair


