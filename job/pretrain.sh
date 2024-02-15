#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/logs/finetune%j.out

# a file for errors
#SBATCH --error=../out/logs/finetune%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis1"

# use GPU
##SBATCH --gpus=geforce:4
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:4

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=64200

# CPU allocated
#SBATCH --cpus-per-task=8

#SBATCH --job-name=med-ssl
#SBATCH --time=8:00:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ai-research-foundations/code

conda activate tf2-gpu

python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"

# ORIGINAL
# python $PROJPATH/run2.py --train_mode=pretrain \
#   --train_batch_size=512 --train_epochs=1000 \
#   --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
#   --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
#   --use_blur=False --color_jitter_strength=0.5 \
#   --model_dir=/tmp/simclr_test --use_tpu=False

python $PROJPATH/run2.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=mimic_cxr --image_size=32 --eval_split=test --train_split=train --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=out/ --use_tpu=False