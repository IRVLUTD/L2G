#!/usr/bin/env bash
set -e




python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000005 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1


python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000006 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1


python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000007 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1


python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000008 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1