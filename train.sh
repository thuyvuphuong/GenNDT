export MODEL_NAME="pretrained_model/stable-diffusion-v1-4"
export DATASET_NAME="datasets/NDTSignal_fulltext"

CUDA_VISIBLE_DEVICES="7" accelerate launch --mixed_precision="fp16"  training.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=0 \
  --train_batch_size=128 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --checkpointing_steps=1000 --checkpoints_total_limit=10 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="genNDT_model"