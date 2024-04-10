import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Specify the URL of the new target webpage
url = "https://dblp.org/db/conf/aaai/aaai2023.html"

# Load the tokenizer and model from Hugging Face Model Hub
model_checkpoint = "Gao-Tianci/RussScholar-Seeker"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)


def predict_russian_author(names):
    # Preparing the model and tokenizer
    model.eval()
    inputs = tokenizer(names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.numpy()


# Fetch and parse the webpage
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
entries = soup.find_all("li", class_="entry inproceedings")

for entry in entries:
    try:
        title = entry.find(class_="title").text
        authors = [span.text for span in entry.find_all("span", itemprop="name")]
        predictions = predict_russian_author(authors)

        # Check if there's at least one Russian author
        if 1 in predictions:
            print(f"Title: {title}")
            print(f"Authors: {', '.join(authors)}")
            russian_authors = [authors[i] for i, pred in enumerate(predictions) if pred == 1]
            print(f"Russian Authors: {', '.join(russian_authors)}")
            print("-" * 60)
    except Exception as e:
        print(f"Error processing entry: {e}")