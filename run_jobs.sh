#!/bin/bash

# model architectures
python TIS_transformer.py train 'data/GRCh38p13/' 'chr11.npy' 'chr4.npy' --gpu '1' --dim 20 --heads 4 --depth 4 --dim_head 10 --max_epochs 100 --local_attn_heads 3 --name 'tis_chr4_small'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr11.npy' 'chr4.npy' --gpu '1' --dim 48 --heads 8 --depth 8 --dim_head 16 --max_epochs 60 --name 'tis_chr4_large'

# cross evaluation 
python TIS_transformer.py train 'data/GRCh38p13/' 'chr8.npy' 'chr1.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr1'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr9.npy' 'chr2.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr2'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr10.npy' 'chr3.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr3'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr11.npy' 'chr4.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr4'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr12.npy' 'chr5.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr5'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr13.npy' 'chr6.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr6'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr14.npy' 'chr7.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr7'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr1.npy' 'chr8.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr8'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr2.npy' 'chr9.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr9'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr3.npy' 'chr10.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr10'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr4.npy' 'chr11.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr11'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr5.npy' 'chr12.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr12'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr6.npy' 'chr13.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr13'
python TIS_transformer.py train 'data/GRCh38p13/' 'chrX.npy' 'chr14.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr14'
python TIS_transformer.py train 'data/GRCh38p13/' 'chrY.npy' 'chr15.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr15'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr19.npy' 'chr16.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr16'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr20.npy' 'chr17.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr17'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr21.npy' 'chr18.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr18'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr22.npy' 'chr19.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr19'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr7.npy' 'chr20.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr20'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr15.npy' 'chr21.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr21'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr16.npy' 'chr22.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chr22'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr17.npy' 'chrX.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chrX'
python TIS_transformer.py train 'data/GRCh38p13/' 'chr18.npy' 'chrY.npy' --gpu '1' --max_epochs 60 --transfer_checkpoint /home/jimc/workspace/TIS_transformer/lightning_logs/mlm_model/version_0/checkpoints/epoch=43-step=660110.ckpt --name 'tis_chrY'