import os
import random
from transformers import BertTokenizer, BertForMaskedLM
import torch

# 设置随机种子以保证实验的可复现性
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 加载BERT tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 设置训练数据路径
data_dir = 'data'

# 读取训练数据
with open(os.path.join(data_dir, 'russian_names.txt'), 'r', encoding='utf-8') as f:
    names = f.readlines()

# 打乱数据
random.shuffle(names)

# 设置训练参数
num_epochs = 3
batch_size = 32
learning_rate = 5e-5

# 将姓名列表转换为输入格式
input_texts = ["[CLS] " + name.strip() + " [SEP]" for name in names]

# 将输入文本转换为token IDs
input_ids = [tokenizer.encode(text, add_special_tokens=False) for text in input_texts]

# 计算输入文本的最大长度
max_len = max(len(ids) for ids in input_ids)

# 将所有输入文本填充到相同长度
input_ids = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]

# 转换为PyTorch张量
input_ids = torch.tensor(input_ids)

# 定义模型优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(0, len(input_ids), batch_size):
        optimizer.zero_grad()
        batch_input_ids = input_ids[i:i+batch_size].to(model.device)
        labels = batch_input_ids.clone()
        labels[labels != tokenizer.mask_token_id] = -100  # 忽略非mask位置的预测
        outputs = model(input_ids=batch_input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# 保存模型
output_dir = 'bert_russian_names_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
