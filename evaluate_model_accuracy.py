import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 设置设备
device = torch.device("cpu")  # 在CPU上运行

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 设置数据路径
data_dir = 'data'
file_name = 'russian_names.csv'  # CSV文件名

# 尝试不同的编码格式来读取文件
encodings = ['utf-8', 'iso-8859-1']

for encoding in encodings:
    try:
        data = pd.read_csv(os.path.join(data_dir, file_name), encoding=encoding)
        break  # 如果成功读取，则跳出循环
    except UnicodeDecodeError:
        continue  # 如果遇到解码错误，则尝试下一个编码格式

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
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载训练好的模型
model_dir = 'bert_russian_names_model'
model = BertForSequenceClassification.from_pretrained(model_dir)

# 评估模型准确性
model.eval()
all_predictions = []
all_labels = []
for batch in val_loader:
    batch = [b.to(device) for b in batch]  # 将张量发送到设备
    inputs, labels = batch
    with torch.no_grad():
        outputs = model(inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# 计算准确率
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {accuracy}")


