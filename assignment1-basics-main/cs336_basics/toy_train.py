from cs336_basics.TransformerLM import Transformer
from cs336_basics.train import train
import torch.nn as nn
import numpy as np
import torch
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

transformer_instance = Transformer(
    d_model=512,          # 模型维度 (一个较小的值)
    num_heads=16,        # 注意力头数 (d_model 必须能被 num_heads 整除, 8 % 2 == 0)
    d_ff=1344,             # 前馈神经网络的隐藏层维度 (可以设为 d_model 或其倍数，这里设为 d_model)
    theta=10000,      # RoPE (旋转位置编码) 的参数 theta，通常是 10000.0
    token_pos=256,       # 这个参数的含义不是很明确，假设它代表最大位置编码数，与 context_length 一致
    vocab_size=10000,      # 词汇表大小 (一个较小的值)
    context_length=256,  # 上下文长度/最大序列长度 (一个较小的值)
    num_layers=4        # Transformer 块的层数 (最小为 1)
)
transformer_instance.to(device= get_device())


toy_long_sequence_data = np.random.randint(0, 30, size=1000, dtype=np.int64)




models_to_train = [
    ("transformer_instance", transformer_instance)
]
for model_name_str, current_model_obj in models_to_train:
# 更新后的 train() 函数调用：
    train(
        data_dir=toy_long_sequence_data,  # 直接传递 NumPy 数组
        model=current_model_obj,             # 训练的模型实例
        batch_size=4,                     # 批处理大小
        learning_rate=0.001,              # 学习率
        epochs=10,                         # 训练轮数
        checkpoint_dir=f"./checkpoints/{model_name_str}",  # 模型检查点保存目录路径 (这个可能仍然是字符串路径)
        beta1=0.9,                        # Adam优化器参数 beta1
        beta2=0.999,                      # Adam优化器参数 beta2
        Lambda=0.01,                      # 正则化系数 Lambda
        eps=1e-8,                         # Adam优化器参数 epsilon
        context_length=256,                 # 上下文长度，用于从长序列中切分数据块
        model_name=model_name_str
    )