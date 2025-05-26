import sys
sys.path.append("/Users/bitkira/Documents/GitHub/Stanford-CS336/assignment1-basics-main/")
from cs336_basics.DataLoader import DataLoader
from cs336_basics.CheckingPoint import save_checkpoint, load_checkpoint
from cs336_basics.CrossEntropy import CrossEntropy
from cs336_basics.AdamW import AdamW
import torch.nn as nn
import argparse
import numpy as np
import wandb
from einops import rearrange

def train(data_dir, model:nn.Module,model_name:str, batch_size, learning_rate, epochs, checkpoint_dir, beta1, beta2, Lambda, eps, context_length, device_string="mps"):
        # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="1120211519-beijing-institute-of-technology",
        # Set the wandb project where this run will be logged.
        project="cs336hw1",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": model_name,
            "dataset": data_dir,
            "beta1": beta1,
            "beta2": beta2,
            "Lambda": Lambda,
            "eps": eps,
            "context_length": context_length,
            "batch_size": batch_size,
        },
    )
    input_token = data_dir
    # input_token = np.memmap(data_dir, dtype=np.float32, mode='r')
    optimizer  = AdamW(model.parameters(), learning_rate, (beta1, beta2), eps, Lambda)
    for i in range(epochs):
        for j in range(len(input_token)//(batch_size*context_length)):
            input, label = DataLoader(input_token, batch_size, context_length, device_string)
            optimizer.zero_grad()
            logits = model(input)
            logits = rearrange(logits, "batchsize seqlen vocabsize -> (batchsize seqlen) vocabsize")
            label = rearrange(label, "batchsize seqlen -> (batchsize seqlen) ")
            loss = CrossEntropy(logits, label)
            loss.backward()
            optimizer.step()
            run.log({"loss": loss})
        save_checkpoint(model, optimizer, i, checkpoint_dir)
    run.finish()

if __name__ == "__name__":
    parser = argparse.ArgumentParser(description="cs336模型训练脚本")
    parser.add_argument('--data_dir', type=str, required=True, help='数据集所在的目录路径')
    parser.add_argument('--model', type=str, required=True, help='要使用的模型架构名称')
    parser.add_argument('--batch_size', type=int, default=32, help='每个批次的样本数量 (默认值: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='优化器的学习率 (默认值: 0.01)')
    parser.add_argument('--beta1', type=float, default=0.01, help='AdamW的超参数Beta1')
    parser.add_argument('--beta2', type=float, default=0.01, help='AdamW的超参数Beta2')
    parser.add_argument('--Lambda', type=float, default=0.01, help='AdamW的超参数Lambda')
    parser.add_argument('--eps', type=float, default=0.003, help='AdamW的超参数伊欧西隆')
    parser.add_argument('--epochs', type=int, default=10, help='训练的总轮数 (默认值: 10)')
    parser.add_argument('--context_length', type=int, default=256, help='文本token长度')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint_dir", help='检查点输出路径')
    parser.add_argument('--device_string', type=str, default="gpu", help='模型训练硬件')

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        beta1=args.beta1,
        beta2=args.beta2,
        Lambda=args.Lambda,
        context_length=args.context_length,
        device_string = args.device_string
    )