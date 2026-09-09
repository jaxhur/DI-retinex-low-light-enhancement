#!/usr/bin/env bash
# 依次训练并测试 LOL-v1、LOL-v2-real、LOL-v2-syn，所有路径相对项目根目录。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 不在脚本中绑定物理显卡；调用方必须在启动命令中明确选择可见单卡。
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "错误：请指定单卡，例如 CUDA_VISIBLE_DEVICES=0 bash run_all_lol_experiments.sh" >&2
  exit 1
fi

require_directory() {
  local directory="$1"
  if [[ ! -d "$directory" ]]; then
    echo "错误：缺少数据目录：$directory" >&2
    exit 1
  fi
}

verify_fresh_checkpoint() {
  local checkpoint="$1"
  local training_started_at="$2"
  local checkpoint_modified_at

  if [[ ! -f "$checkpoint" ]]; then
    echo "错误：训练结束后未找到 checkpoint：$checkpoint" >&2
    exit 1
  fi

  checkpoint_modified_at="$(stat -c %Y "$checkpoint")"
  if [[ "$checkpoint_modified_at" -lt "$training_started_at" ]]; then
    echo "错误：$checkpoint 不是本次训练生成的最新权重，已停止测试。" >&2
    exit 1
  fi
}

# 在耗时训练前一次性检查所有 train/test 数据目录，避免中途才发现路径错误。
require_directory "data/LOL-v1/our485/low"
require_directory "data/LOL-v1/eval15/low"
require_directory "data/LOL-v1/eval15/high"
require_directory "data/LOL-v2/Real_captured/Train/Low"
require_directory "data/LOL-v2/Real_captured/Test/Low"
require_directory "data/LOL-v2/Real_captured/Test/Normal"
require_directory "data/LOL-v2/Synthetic/Train/Low"
require_directory "data/LOL-v2/Synthetic/Test/Low"
require_directory "data/LOL-v2/Synthetic/Test/Normal"

echo "使用 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "[1/3] 训练 LOL-v1"
mkdir -p models/di_retinex_lolv1
LOLV1_STARTED_AT="$(date +%s)"
python -u lowlight_train.py \
  --lowlight_images_path "data/LOL-v1/our485/low" \
  --num_epochs 1000 \
  --train_batch_size 1 \
  --snapshot_epoch 10 \
  --snapshots_folder models/di_retinex_lolv1/
verify_fresh_checkpoint "models/di_retinex_lolv1/latest.pth" "$LOLV1_STARTED_AT"

echo "[1/3] 测试 LOL-v1"
python -u lowlight_test.py \
  --lowlight_images_path "data/LOL-v1/eval15/low" \
  --gt_images_path "data/LOL-v1/eval15/high" \
  --model_path models/di_retinex_lolv1/latest.pth \
  --save_path test_result/di_retinex_lolv1/LOL-v1/enhanced \
  --experiment_name di_retinex_lolv1 \
  --dataset_name LOL-v1 \
  --out_ch 6 \
  --inner_ch 64

echo "[2/3] 训练 LOL-v2-real"
mkdir -p models/di_retinex_lolv2_real
LOLV2_REAL_STARTED_AT="$(date +%s)"
python -u lowlight_train.py \
  --lowlight_images_path "data/LOL-v2/Real_captured/Train/Low" \
  --num_epochs 1000 \
  --train_batch_size 1 \
  --snapshot_epoch 10 \
  --snapshots_folder models/di_retinex_lolv2_real/
verify_fresh_checkpoint "models/di_retinex_lolv2_real/latest.pth" "$LOLV2_REAL_STARTED_AT"

echo "[2/3] 测试 LOL-v2-real"
python -u lowlight_test.py \
  --lowlight_images_path "data/LOL-v2/Real_captured/Test/Low" \
  --gt_images_path "data/LOL-v2/Real_captured/Test/Normal" \
  --model_path models/di_retinex_lolv2_real/latest.pth \
  --save_path test_result/di_retinex_lolv2_real/LOL-v2-real/enhanced \
  --experiment_name di_retinex_lolv2_real \
  --dataset_name LOL-v2-real \
  --out_ch 6 \
  --inner_ch 64

echo "[3/3] 训练 LOL-v2-syn"
mkdir -p models/di_retinex_lolv2_syn
LOLV2_SYN_STARTED_AT="$(date +%s)"
python -u lowlight_train.py \
  --lowlight_images_path "data/LOL-v2/Synthetic/Train/Low" \
  --num_epochs 1000 \
  --train_batch_size 1 \
  --snapshot_epoch 10 \
  --snapshots_folder models/di_retinex_lolv2_syn/
verify_fresh_checkpoint "models/di_retinex_lolv2_syn/latest.pth" "$LOLV2_SYN_STARTED_AT"

echo "[3/3] 测试 LOL-v2-syn"
python -u lowlight_test.py \
  --lowlight_images_path "data/LOL-v2/Synthetic/Test/Low" \
  --gt_images_path "data/LOL-v2/Synthetic/Test/Normal" \
  --model_path models/di_retinex_lolv2_syn/latest.pth \
  --save_path test_result/di_retinex_lolv2_syn/LOL-v2-syn/enhanced \
  --experiment_name di_retinex_lolv2_syn \
  --dataset_name LOL-v2-syn \
  --out_ch 6 \
  --inner_ch 64

echo "全部三组 LOL 训练与测试已完成。"
