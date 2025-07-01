#!/bin/bash

# 快速训练tokenizer脚本
echo "开始快速tokenizer训练..."
echo "使用内存映射加速文本加载"

# 设置环境变量
export PYTHONPATH="/home/overman/workspaces/stanford-cs336/minbpe:$PYTHONPATH"

# 运行快速训练
nohup /home/overman/workspaces/stanford-cs336/minbpe/.conda/bin/python /home/overman/workspaces/stanford-cs336/minbpe/train_tokenizer_fast.py > fast_training.log 2>&1 &

echo "训练任务已在后台启动"
echo "日志文件: fast_training.log"
echo "查看进度: tail -f fast_training.log" 