#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=out/logs/%j.out

# a file for errors
#SBATCH --error=out/logs/%j.err

# select the node edith
#SBATCH --nodelist="edith1"
#SBATCH --partition=edith

# use one 2080Ti GPU
#SBATCH --gpus=geforce:1

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=16384

# CPU allocated
#SBATCH --cpus-per-task=1

#SBATCH --job-name=remedis
#SBATCH --time=5:00

source /ubc/cs/research/shield/projects/payamfz/env/bin/activate

python medical-ai-research-foundations-framework/medical-ai-research-foundations/run.py --train_mode=pretrain \
  --train_batch_size=10 --train_epochs=2 \
  --learning_rate=0.3 --weight_decay=1e-6 --temperature=0.1 \
  --dataset=cifar10 --image_size=224 --eval_split=test --resnet_depth=18 \
  --use_blur=True --color_jitter_strength=1.0 \
  --model_dir=../out/my-models --use_tpu=False

deactivate