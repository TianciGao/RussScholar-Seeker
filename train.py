import os
import random
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split

# 设置随机种子以保证实验的可复现性
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 设置数据路径
data_dir = 'data'
file_name = 'russian_names.csv'  # CSV文件名

# 使用pandas读取CSV文件，尝试不同的编码方式处理编码问题
encodings = ['utf-8', 'latin1']
for encoding in encodings:
    try:
        file_path = os.path.join(data_dir, file_name)
        data = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        print(f"Error: Unable to decode using {encoding} encoding. Trying another encoding...")

# 如果所有编码方式都失败了，则输出错误信息并退出程序
else:
    print("Error: Unable to decode file using any encoding.")
    exit()

names = data['name'].tolist()
labels = data['label'].tolist()

# 数据预处理，分割为训练集和测试集
train_texts, val_texts, train_labels, val_labels = train_test_split(names, labels, test_size=.1)

# 将文本和标签编码
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 转换为torch张量
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_labels))
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']), torch.tensor(val_labels))

# 初始化模型，对于二分类任务
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # 将数据移到相同的设备上
        batch = [b.to(device) for b in batch]
        inputs, labels = batch
        model.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# 保存模型
output_dir = 'bert_russian_names_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


