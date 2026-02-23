#!/usr/bin/env bash
set -e

# python sam2_trainer.py \
#   --data_root ../SSD2/High_datasets \
#   --object_id 000013 \
#   --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#   --checkpoint checkpoints/sam2.1_hiera_large.pt \
#   --epochs 12 \
#   --lr 5e-3 \
#   --device cuda:1

# python sam2_trainer.py \
#   --data_root ../SSD2/High_datasets \
#   --object_id 000015 \
#   --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#   --checkpoint checkpoints/sam2.1_hiera_large.pt \
#   --epochs 12 \
#   --lr 5e-3 \
#   --device cuda:1


# python sam2_trainer.py \
#   --data_root ../SSD2/High_datasets \
#   --object_id 000017 \
#   --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#   --checkpoint checkpoints/sam2.1_hiera_large.pt \
#   --epochs 12 \
#   --lr 5e-3 \
#   --device cuda:1


# python sam2_trainer.py \
#   --data_root ../SSD2/High_datasets \
#   --object_id 000018 \
#   --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#   --checkpoint checkpoints/sam2.1_hiera_large.pt \
#   --epochs 12 \
#   --lr 5e-3 \
#   --device cuda:1

# python sam2_trainer.py \
#   --data_root ../SSD2/High_datasets \
#   --object_id 000019 \
#   --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
#   --checkpoint checkpoints/sam2.1_hiera_large.pt \
#   --epochs 12 \
#   --lr 5e-3 \
#   --device cuda:1


python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000012 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1



python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000011 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1



python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000016 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1

python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000020 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1

python sam2_trainer.py \
  --data_root ../SSD2/High_datasets \
  --object_id 000014 \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --epochs 12 \
  --lr 5e-3 \
  --device cuda:1