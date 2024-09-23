import requests
import bibtexparser
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

model_checkpoint = "Gao-Tianci/RussScholar-Seeker"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

def process_entry(title, authors, doi):
    predictions = predict_russian_author(authors)
    if 1 in predictions:
        russian_authors = [authors[i] for i, pred in enumerate(predictions) if pred == 1]
        print(f"Title: {title}")
        print(f"Authors: {', '.join(authors)}")
        print(f"Russian Authors: {', '.join(russian_authors)}")
        print(f"DOI: {doi}")
        print("-" * 60)

def detect_content_type(response_headers):
    content_type = response_headers.get('Content-Type', '')
    if 'xml' in content_type:
        return 'xml'
    elif 'bibtex' in content_type:
        return 'bibtex'
    else:
        return 'html'

# URL for testing
url = "https://dblp.org/search/publ/api?q=stream%3Astreams%2Fconf%2Faaai%3A&h=1000&format=xml"
response = requests.get(url)
content_type = detect_content_type(response.headers)


start_time = time.time()  # Start time of the program
if content_type == 'html':
    parse_html(response.text)
elif content_type == 'xml':
    parse_xml(response.text)
elif content_type == 'bibtex':
    parse_bibtex(response.text)
print(f"Total processing time: {time.time() - start_time} seconds")  # End time of the program
