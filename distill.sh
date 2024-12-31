export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATAPATH="/path/to/t2i_1024"
export OUTPUT_DIR="ckpt/training_exp"
export PRECISION="bf16"

accelerate launch --config_file deepspeed_config.yaml distill.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
  --data_root=$DATAPATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision=$PRECISION \
  --dataloader_num_workers=8 \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --validation_epochs=1 \
  --seed="0" \
  --checkpointing_steps=5000 \
  --use_cached_prompt_embed \
  --use_cached_latent \
  --gradient_checkpointing \
  --down_factor=1 \
  --window_size=16
  
