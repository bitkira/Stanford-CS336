import torch
import numpy as np
import os
import array # 用于创建二进制文件
from torch.utils.data import Dataset, DataLoader

# --------------- 1. 定义高效的 Dataset 类 (与之前一致) ---------------
class MemmapTokenDataset(Dataset):
    def __init__(self, token_file_path, dtype_np, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        
        if not os.path.exists(token_file_path):
            raise FileNotFoundError(f"Token文件未找到: {token_file_path}")

        file_size_bytes = os.path.getsize(token_file_path)
        # 使用 NumPy 的 dtype 对象来获取 itemsize，更可靠
        try:
            self.dtype_np_obj = np.dtype(dtype_np)
        except TypeError:
            raise TypeError(f"提供的 dtype_np '{dtype_np}' 不是有效的 NumPy 数据类型。请使用例如 np.uint16, np.int32 等。")
            
        item_size_bytes = self.dtype_np_obj.itemsize

        if file_size_bytes == 0:
            raise ValueError(f"Token文件为空: {token_file_path}")
        if file_size_bytes % item_size_bytes != 0:
            raise ValueError(f"Token文件大小 {file_size_bytes} 不是单个条目大小 {item_size_bytes} (dtype: {self.dtype_np_obj.name}) 的整数倍。")
            
        num_tokens = file_size_bytes // item_size_bytes
        
        # 使用 NumPy 的 dtype 对象打开 memmap
        self.token_data = np.memmap(token_file_path, dtype=self.dtype_np_obj, mode='r', shape=(num_tokens,))
        print(f"MemmapTokenDataset: 成功加载 {len(self.token_data)} 个类型为 {self.dtype_np_obj.name} 的 tokens from {token_file_path}")
        
        if len(self.token_data) <= self.sequence_length:
            raise ValueError(f"Token序列总长度 ({len(self.token_data)}) "
                             f"必须大于设定的序列长度 ({self.sequence_length}).")
        self.num_samples = len(self.token_data) - self.sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"索引 {idx} 超出范围 (可用样本数: {self.num_samples}).")
            
        input_tokens_np = self.token_data[idx : idx + self.sequence_length]
        target_tokens_np = self.token_data[idx + 1 : idx + self.sequence_length + 1]
        
        # 返回 PyTorch long tensors (通常嵌入层需要LongTensor)
        return torch.tensor(input_tokens_np, dtype=torch.long), torch.tensor(target_tokens_np, dtype=torch.long)

# --------------- 2. 创建一个虚拟的 .bin token 文件用于演示 ---------------
def create_dummy_token_file(file_path, num_tokens, vocab_size, token_storage_type_char='H'):
    """
    创建一个包含随机token ID的虚拟二进制文件。
    'H' -> unsigned short (2 bytes), 'I' -> unsigned int (4 bytes)
    """
    print(f"正在创建虚拟token文件: {file_path}...")
    if os.path.exists(file_path):
        print("文件已存在，将重新创建。")
        os.remove(file_path)
        
    # 生成随机 token ID (范围在 [0, vocab_size-1])
    dummy_tokens = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16 if token_storage_type_char == 'H' else np.uint32)
    
    with open(file_path, "wb") as f:
        id_array = array.array(token_storage_type_char, dummy_tokens)
        id_array.tofile(f)
    print(f"虚拟token文件创建完毕，包含 {num_tokens} 个tokens，词汇表大小模拟为 {vocab_size}。")
    print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")

# --------------- 3. 主程序：实例化和使用 DataLoader ---------------
if __name__ == "__main__":
    # --- 配置参数 ---
    dummy_file_path = "./dummy_tokens.bin"
    num_dummy_tokens = 1_000_000  # 创建100万个token的虚拟文件
    dummy_vocab_size = 50257     # 模拟类似GPT-2的词汇表大小
    
    # 与 create_dummy_token_file 中的 token_storage_type_char 对应
    # 'H' 对应 np.uint16 (如果 vocab_size <= 65535)
    # 'I' 对应 np.uint32
    # 这里我们用 'H' 因为 dummy_vocab_size < 65535
    dummy_token_storage_char = 'H' 
    dtype_for_memmap = np.uint16 # 这个必须和 dummy_token_storage_char 精确对应

    sequence_length = 256      # 输入给模型的序列长度
    batch_size = 32            # 批量大小
    num_epochs_to_iterate = 1  # 演示时迭代的epoch次数
    
    # 决定使用哪个设备
    device_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    print(f"将使用设备: {device}")

    # --- 创建虚拟文件 ---
    create_dummy_token_file(dummy_file_path, num_dummy_tokens, dummy_vocab_size, dummy_token_storage_char)

    # --- 实例化 Dataset ---
    print("\n正在实例化 Dataset...")
    try:
        train_dataset = MemmapTokenDataset(
            token_file_path=dummy_file_path,
            dtype_np=dtype_for_memmap,      # 确保这个dtype与创建文件时使用的精确匹配
            sequence_length=sequence_length
        )
    except Exception as e:
        print(f"创建 Dataset 失败: {e}")
        exit()

    # --- 实例化 DataLoader ---
    print("\n正在实例化 DataLoader...")
    # num_workers > 0 会启用多进程数据加载，提高效率
    # pin_memory=True 与 non_blocking=True (在 .to(device) 中使用) 配合，可以加速CPU到GPU的数据传输
    num_loader_workers = 4 if device.type == 'cuda' else 0 # CPU训练时通常不需要或用处不大
    
    # 确保在Windows上或使用'spawn'/'forkserver'方法时，多进程代码在 if __name__ == '__main__': 内
    # 对于Linux的默认'fork'方法，通常没问题。
    if os.name == 'posix' and num_loader_workers > 0 : # 简单示例，更复杂的判断可能需要
        print(f"在类Unix系统上，将使用 {num_loader_workers} 个 worker。")
    elif num_loader_workers > 0:
        print(f"警告：在非类Unix系统上使用 num_workers > 0 可能需要特殊处理或可能导致问题，"
              f"当前设置为 {num_loader_workers}。如果遇到问题，尝试设为0。")
              
    actual_pytorch_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时通常需要打乱数据
        num_workers=num_loader_workers,
        pin_memory=(device.type == 'cuda'), 
        drop_last=True # 如果最后一个批次数据不完整，则丢弃它
    )
    print(f"DataLoader 实例化完毕。总样本数: {len(train_dataset)}, "
          f"每个epoch的批次数: {len(actual_pytorch_dataloader)}")

    # --- 演示如何从 DataLoader 中迭代获取数据 ---
    print("\n开始从 DataLoader 中迭代数据批次...")
    for epoch in range(num_epochs_to_iterate):
        print(f"\n--- Epoch {epoch+1}/{num_epochs_to_iterate} ---")
        for i, (input_batch, label_batch) in enumerate(actual_pytorch_dataloader):
            # input_batch 和 label_batch 是从 Dataset 的 __getitem__ 返回的，默认在CPU上
            # 现在需要将它们移动到目标设备
            input_data = input_batch.to(device, non_blocking=True if device.type == 'cuda' and num_loader_workers > 0 and train_dataset else False)
            label_data = label_batch.to(device, non_blocking=True if device.type == 'cuda' and num_loader_workers > 0 and train_dataset else False)

            if i < 3: # 只打印前3个批次的信息作为演示
                print(f"  批次 {i+1}:")
                print(f"    Input batch shape: {input_data.shape}, dtype: {input_data.dtype}, device: {input_data.device}")
                print(f"    Label batch shape: {label_data.shape}, dtype: {label_data.dtype}, device: {label_data.device}")
                # print(f"    Input sample (first in batch, first 10 tokens): {input_data[0, :10]}")
                # print(f"    Label sample (first in batch, first 10 tokens): {label_data[0, :10]}")
            elif i == 3 and len(actual_pytorch_dataloader) > 3 :
                 print("    ...") # 表示后续还有更多批次
            
            # 在这里，您可以将 input_data 和 label_data 输入到您的模型进行训练
            # 例如:
            # optimizer.zero_grad()
            # logits = model(input_data)
            # loss = loss_fn(logits.view(-1, logits.size(-1)), label_data.view(-1))
            # loss.backward()
            # optimizer.step()

        print(f"Epoch {epoch+1} 数据迭代完毕。")

    # --- 清理虚拟文件 (可选) ---
    # print(f"\n正在删除虚拟token文件: {dummy_file_path}")
    # os.remove(dummy_file_path)
    print("\n演示结束。")