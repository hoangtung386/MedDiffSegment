#!/bin/bash
# Dual RTX 3090 Optimized Training Script

# GPU optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1  # Both GPUs

# Paths
DATA_DIR="/home/admin1/Documents/data/training"
OUT_DIR="./output_2x3090_$(date +%Y%m%d_%H%M%S)"

# Training params optimized for 2x RTX 3090 (48GB total)
IMAGE_SIZE=256
BATCH_SIZE=32  # 16 per GPU × 2 GPUs - OPTIMAL!
NUM_CHANNELS=128
MAX_STEPS=150000

echo "========================================="
echo "Dual RTX 3090 Training Configuration"
echo "========================================="
echo "GPUs: 2x RTX 3090 (48GB VRAM total)"
echo "Batch Size: $BATCH_SIZE (16 per GPU)"
echo "Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Max Steps: $MAX_STEPS"
echo "Expected time: ~45-55 hours (~2 days) for 150K steps"
echo "Speed: ~2,500-3,500 steps/hour"
echo "Advantage: 1.5-2x faster + better convergence!"
echo "========================================="
echo ""

# Check GPU status before starting
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Start training with multi-GPU
python scripts/train.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --image_size "$IMAGE_SIZE" \
  --num_channels "$NUM_CHANNELS" \
  --class_cond False \
  --num_res_blocks 2 \
  --num_heads 1 \
  --learn_sigma True \
  --use_scale_shift_norm False \
  --attention_resolutions 16 \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --rescale_learned_sigmas False \
  --rescale_timesteps False \
  --lr 1e-4 \
  --batch_size "$BATCH_SIZE" \
  --multi_gpu "0,1" \
  --log_interval 50 \
  --save_interval 2000 \
  --lr_anneal_steps "$MAX_STEPS" \
  2>&1 | tee "$OUT_DIR/training.log"

echo ""
echo "-----------------------------------------"
echo "Training completed at $(date)"
echo "-----------------------------------------"
