#!/bin/bash
set -e

echo "========== CLPsych 2026 Task 3 Pipeline =========="

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

for f in "train_task3.json" "test_task3.json" "task1_pred.json" "task2_pred.json"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] 缺少文件: $f"
        exit 1
    fi
done

for d in "task12_train_data" "task12_test_data"; do
    if [ ! -d "$d" ]; then
        echo "[ERROR] 缺少目录: $d"
        exit 1
    fi
done

echo "[1/3] Training..."
python train.py

echo "[2/3] Inference..."
python inference.py

echo "[3/3] Signatures..."
python task3b.py

echo "[Done]"
echo "   outputs/task3_pred.json"
echo "   outputs/TEAMNAME_Task3b.json"