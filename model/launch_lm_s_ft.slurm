#!/bin/bash
#SBATCH --job-name=lm_filter
#SBATCH --partition=SC-A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=admin
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodelist=angular-001

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

## distributed debug
# export TORCH_CPP_DISTRIBUTED_DEBUG="INFO"
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# export NCCL_DEBUG="INFO"
# export NCCL_P2P_LEVEL="NVL"
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export TXT_PATH="/local/filter.ft"
export LOC_PATH="/local/filter.tell"
export AUX_PATH="/local/aux.pkl"
export OUTPUT_DIR="./lm_ft_s_logs/"
export MODE="lm"
export MODEL_PATH="./deepseek-coder-1.3b-tg-lm-pt-s"

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
     --nproc_per_node $GPUS_PER_NODE \
     --nnodes $SLURM_NNODES \
     --node_rank $SLURM_PROCID \
     --master_addr $MASTER_ADDR \
     --master_port $MASTER_PORT \
     ./geometry-gen/model/trainer.py \
     --txt_path $TXT_PATH \
     --loc_path $LOC_PATH \
     --output_dir $OUTPUT_DIR \
     --mode $MODE \
     --model_name_or_path $MODEL_PATH \
     --seed 117 \
     --weigh_classes False \
     --optim "adamw_torch_fused" \
     --pretrain False \
     --max_steps 10000 \
     --model_max_length 512 \
     --per_device_train_batch_size 64 \
     --per_device_eval_batch_size 128 \
     --gradient_accumulation_steps 4 \
     --dataloader_num_workers 4 \
     --eval_strategy "no" \
     --eval_steps 5000 \
     --save_strategy "steps" \
     --save_steps 100 \
     --save_total_limit 10 \
     --learning_rate 2.5e-5 \
     --warmup_steps 10 \
     --logging_steps 1 \
     --lr_scheduler_type "cosine" \
     --gradient_checkpointing True \
     --report_to "tensorboard" \
     --bf16 True'
