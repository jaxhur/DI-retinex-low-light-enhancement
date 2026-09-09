DI-Retinex自监督训练：训练阶段只读取低照图像，不读取配对 GT；`high` / `Normal` 仅用于测试阶段的 PSNR、RGB SSIM、LPIPS 计算。

论文训练配置为 batch size 1、整图训练、1000 epoch；本地训练脚本已统一默认使用 1000 epoch。

| 数据集 | 训练图像数 | batch size | epoch | 每 epoch update | 总 update / iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| LOL-v1 | 485 | 1 | 1000 | 485 | 485,000 |
| LOL-v2-real | 689 | 1 | 1000 | 689 | 689,000 |
| LOL-v2-syn | 900 | 1 | 1000 | 900 | 900,000 |

每隔 `snapshot_epoch`（默认 100）保存一次 `Epoch*.pth`，并覆盖 `latest.pth`。因此，1000 epoch 时会保存 `Epoch99.pth` 至 `Epoch999.pth`，其中 `latest.pth` 对应最后一次（`Epoch999.pth`）权重，而不是经验证集挑选的最佳权重。



# 创建环境

```bash
conda create -n di-retinex python=3.8 -y
conda activate di-retinex

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111

python -m pip install numpy==1.21.6 Pillow opencv-python natsort lpips==0.1.4 gdown
```



# 下载数据集

```bash
pip install -U gdown
apt install -y unzip

cd ./datasets
# LOL-v1
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
# LOL-v2
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"


# 解压
unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2

rm LOL-v1.zip LOL-v2-renamed.zip
cd ../
```





# LOLv1



LOL-v1：485,000 次 update

```bash
mkdir -p snapshots/di_retinex_lolv1

python lowlight_train.py \
  --lowlight_images_path "data/LOL-v1/our485/low" \
  --num_epochs 1000 \
  --train_batch_size 1 \
  --snapshots_folder snapshots/di_retinex_lolv1/
```

### LOL-v2-real：689,000 次 update

```bash
cd "$PROJECT_ROOT"
mkdir -p snapshots/di_retinex_lolv2_real

python lowlight_train.py \
  --lowlight_images_path "$DATA_ROOT/LOL-v2/Real_captured/Train/Low" \
  --num_epochs 1000 \
  --train_batch_size 1 \
  --snapshots_folder snapshots/di_retinex_lolv2_real/
```

### LOL-v2-syn：900,000 次 update

```bash
cd "$PROJECT_ROOT"
mkdir -p snapshots/di_retinex_lolv2_syn

python lowlight_train.py \
  --lowlight_images_path "$DATA_ROOT/LOL-v2/Synthetic/Train/Low" \
  --num_epochs 1000 \
  --train_batch_size 1 \
  --snapshots_folder snapshots/di_retinex_lolv2_syn/
```

训练超参数沿用脚本默认值：Adam、`lr=0.001`、`weight_decay=0.0001`、`grad_clip_norm=0.1`、`num_workers=4`、整图输入。数据集图像尺寸不同或显存不足时，先调整 `--num_workers`，不要擅自把单卡 batch size 改大后仍把 update 次数当作上述数值。

## 4. 测试与统一指标

测试脚本会递归读取 Low 图，按规范化相对路径与 GT 严格配对，保留相对目录保存增强图。传入 `--gt_images_path` 后，会逐图计算以下指标，并对完整测试集做算术平均：

- PSNR：BasicSR RGB，`crop_border=0`；
- RGB SSIM：BasicSR RGB 三通道平均，`crop_border=0`；
- LPIPS：AlexNet v0.1，RGB 输入范围 `[-1, 1]`。

PSNR 与 RGB SSIM 使用项目内置的 BasicSR 等价最小实现，只保留对应公式和必要的 `numpy`、`opencv-python` 依赖；**无需安装 BasicSR**。

评测不会 resize 图像，也不会使用 GT-Mean。结果图片保存在 `enhanced/`，汇总文件自动写到其父目录的 `metric.csv`。以下命令使用本节训练产生的最终 `latest.pth`；正常模型固定传入 `--out_ch 6 --inner_ch 64`。

### LOL-v1

```bash
cd "$PROJECT_ROOT"

python lowlight_test.py \
  --lowlight_images_path "$DATA_ROOT/LOL-v1/eval15/low" \
  --gt_images_path "$DATA_ROOT/LOL-v1/eval15/high" \
  --model_path snapshots/di_retinex_lolv1/latest.pth \
  --save_path test_result/di_retinex_lolv1/LOL-v1/enhanced \
  --experiment_name di_retinex_lolv1 \
  --dataset_name LOL-v1 \
  --out_ch 6 \
  --inner_ch 64
```

### LOL-v2-real

```bash
python lowlight_test.py \
  --lowlight_images_path "$DATA_ROOT/LOL-v2/Real_captured/Test/Low" \
  --gt_images_path "$DATA_ROOT/LOL-v2/Real_captured/Test/Normal" \
  --model_path snapshots/di_retinex_lolv2_real/latest.pth \
  --save_path test_result/di_retinex_lolv2_real/LOL-v2-real/enhanced \
  --experiment_name di_retinex_lolv2_real \
  --dataset_name LOL-v2-real \
  --out_ch 6 \
  --inner_ch 64
```

### LOL-v2-syn（扩展实验）

```bash
python lowlight_test.py \
  --lowlight_images_path "$DATA_ROOT/LOL-v2/Synthetic/Test/Low" \
  --gt_images_path "$DATA_ROOT/LOL-v2/Synthetic/Test/Normal" \
  --model_path snapshots/di_retinex_lolv2_syn/latest.pth \
  --save_path test_result/di_retinex_lolv2_syn/LOL-v2-syn/enhanced \
  --experiment_name di_retinex_lolv2_syn \
  --dataset_name LOL-v2-syn \
  --out_ch 6 \
  --inner_ch 64
```

如需先验证仓库随附的预训练权重，`weights/latest (21.54 lolv1).pth` 对应 LOL-v1 正常模型，`weights/latest (lolv2 21.35).pth` 对应 LOL-v2-real 正常模型；将上面对应测试命令的 `--model_path` 替换为带引号的权重路径即可。LOL-v2-syn 没有仓库随附的对应权重，应使用其独立训练结果。

## 5. 输出位置与结果说明

每一组训练得到如下权重文件：

```text
snapshots/di_retinex_lolv1/Epoch99.pth ... Epoch999.pth
snapshots/di_retinex_lolv1/latest.pth
snapshots/di_retinex_lolv2_real/Epoch99.pth ... Epoch999.pth
snapshots/di_retinex_lolv2_real/latest.pth
snapshots/di_retinex_lolv2_syn/Epoch99.pth ... Epoch999.pth
snapshots/di_retinex_lolv2_syn/latest.pth
```

三套测试输出分别为：

```text
test_result/di_retinex_lolv1/LOL-v1/enhanced/
test_result/di_retinex_lolv1/LOL-v1/metric.csv
test_result/di_retinex_lolv2_real/LOL-v2-real/enhanced/
test_result/di_retinex_lolv2_real/LOL-v2-real/metric.csv
test_result/di_retinex_lolv2_syn/LOL-v2-syn/enhanced/
test_result/di_retinex_lolv2_syn/LOL-v2-syn/metric.csv
```

`metric.csv` 记录实验名、数据集名、图像数、平均 PSNR、`ssim_mode=RGB`、平均 LPIPS、`lpips_backbone=alex`、`lpips_version=0.1`、checkpoint 和增强图目录。未传入 `--gt_images_path` 时，脚本仍可只保存增强图，但不会产生全参考指标或 `metric.csv`。

