import requests
import bibtexparser
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import pandas as pd

model_checkpoint = "Gao-Tianci/RussScholar-Seeker"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

entries_data = []  # 用于存储所有条目的信息

def predict_russian_author(names):
    inputs = tokenizer(names, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return predictions

def format_doi(doi):
    if doi and not doi.startswith('https://'):
        return f"https://doi.org/{doi}"
    return doi

def process_entry(title, authors, doi):
    predictions = predict_russian_author(authors)
    if 1 in predictions:
        russian_authors = [authors[i] for i, pred in enumerate(predictions) if pred == 1]
        entry_info = {
            "Title": title,
            "Authors": ', '.join(authors),
            "Russian Authors": ', '.join(russian_authors),
            "DOI": doi
        }
        entries_data.append(entry_info)

def save_to_excel():
    df = pd.DataFrame(entries_data)
    df.to_excel("output.xlsx", index=False)

# 以下是具体的解析函数和内容类型检测函数
def parse_xml(content):
    root = ET.fromstring(content)
    for hit in root.findall('.//hit'):
        title = hit.find('.//title').text
        authors = [author.text for author in hit.findall('.//author')]
        doi = hit.find('.//doi').text if hit.find('.//doi') is not None else 'DOI not available'
        doi = format_doi(doi)
        process_entry(title, authors, doi)

def parse_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    entries = soup.find_all("li", class_="entry inproceedings")
    for entry in entries:
        title = entry.find(class_="title").text
        authors = [span.text for span in entry.find_all("span", itemprop="name")]
        doi_element = entry.find("a", href=lambda href: href and "doi.org" in href)
        doi = doi_element["href"] if doi_element else 'DOI not available'
        doi = format_doi(doi)
        process_entry(title, authors, doi)

def parse_bibtex(content):
    bib_database = bibtexparser.loads(content)
    for entry in bib_database.entries:
        title = entry['title']
        authors = entry['author'].split(' and ')
        doi = entry.get('doi', 'DOI not available')
        doi = format_doi(doi)
        process_entry(title, authors, doi)

def detect_content_type(response_headers):
    content_type = response_headers.get('Content-Type', '')
    if 'xml' in content_type:
        return 'xml'
    elif 'bibtex' in content_type:
        return 'bibtex'
    else:
        return 'html'

# 主程序
url = "https://dblp.org/db/conf/aaai/aaai2021.html"
response = requests.get(url)
content_type = detect_content_type(response.headers)

start_time = time.time()
if content_type == 'html':
    parse_html(response.text)
elif content_type == 'xml':
    parse_xml(response.text)
elif content_type == 'bibtex':
    parse_bibtex(response.text)


# 保存到Excel
save_to_excel()
print(f"Total processing time: {time.time() - start_time} seconds")
