import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch


class RussianNamesDataset(Dataset):
    def __init__(self, names, tokenizer, max_length=128):
        self.names = names
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        encoding = self.tokenizer.encode_plus(
            name,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


# Set the device to GPU (cuda) if available, otherwise stick with CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to your locally saved model and tokenizer
model_path = 'D:/program/Code/RussScholar-Seeker/bert_russian_names_model'

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Move model to the correct device
model = model.to(device)
model.eval()

# Load the test dataset
data_dir = 'D:/program/Code/RussScholar-Seeker/data'
file_name = 'russian_names.csv'
file_path = os.path.join(data_dir, file_name)
test_data = pd.read_csv(file_path, encoding='latin1')
test_names = test_data['name'].tolist()

# Create dataset and dataloader
test_dataset = RussianNamesDataset(test_names, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model
with torch.no_grad():
    predictions = []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Assuming binary classification with {0: "Non-Russian", 1: "Russian"}
predictions = ["Russian" if pred == 1 else "Non-Russian" for pred in predictions]

# Output predictions in sentence form
for name, prediction in zip(test_names, predictions):
    print(f"{name} is a {prediction} name.")

