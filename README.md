**RussScholar-Seeker：A Python package for predicting whether a name is Russian**

I am aware that this topic may be viewed from a political perspective. That is absolutely AGAINST my motivation.

<p align="center"><img width="410" alt="微信图片_20240409101854" src="https://github.com/TianciGao/RussScholar-Seeker/assets/153629778/ba73d175-091d-4439-925c-82cd9a97f7b9">

This project contains a series of programs designed to automatically identify and analyze Russian authors in academic papers. Utilizing the latest natural language processing technologies, it predicts the geographical attribute of names using a pre-trained BERT model to determine whether a given name is Russian.

## Principle

The core of the project is based on the `BertForSequenceClassification` model from the `transformers` library, trained with a specific dataset to distinguish Russian from non-Russian names. We first scrape metadata of academic papers, including titles and author names, from databases like DBLP. Then, we use this trained model to predict the names fetched, automatically identifying Russian authors.
https://huggingface.co/Gao-Tianci/RussScholar-Seeker

## Production Process

1. **Data Preparation**: First, we collected a set of names labeled as Russian and non-Russian to serve as the dataset for training the model.
2. **Model Training**: We trained the model using `BertForSequenceClassification` and the collected dataset. During the training process, we adjusted the model parameters to achieve the best predictive performance.
3. **Data Scraping**: We wrote web scraping programs to fetch metadata of academic papers from databases like DBLP.
4. **Prediction and Analysis**: The fetched names were predicted using the trained model to identify Russian authors, and the related information was output.

## Usage Guide

Before using this tool, you need to install some necessary Python libraries, including `transformers`, `torch`, `requests`, and `beautifulsoup4`. The installation command is as follows:

```bash

pip install transformers torch requests beautifulsoup4
```
After that, you can run prediction.py to execute the Russian expert identification. The command might look like this:

```bash
python prediction.py
```
## Case Study: Identifying Russian Authors in AAAI 2021

One of the notable applications of this project was the analysis of academic papers from the AAAI 2021 conference, listed on DBLP(HTML,XML). The goal was to identify papers with Russian authors, showcasing the model's ability to provide insights into geographical distributions of academic contributions.

### Results
<p align="center"><img width="615" alt="1712759742827" src="https://github.com/TianciGao/RussScholar-Seeker/assets/153629778/2cd01309-38cc-4fab-a9bc-8316c023e69f">

The model successfully identified several papers with Russian authors, underlining the global collaboration in the field of Artificial Intelligence. Here are a few highlights from the analysis:

<p align="center"><img width="828" alt="1712759861119" src="https://github.com/TianciGao/RussScholar-Seeker/assets/153629778/fed0c4ee-b3f0-4452-b92b-e3fac6098ac8">


These results not only demonstrate the practical utility of the Russian Expert Identifier in analyzing academic contributions but also highlight the diverse international collaboration within the AI research community.

### Implications

This case study underscores the potential of AI and NLP technologies in enhancing our understanding of academic landscapes. By automating the identification of geographical attributes of authors, we can gain valuable insights into global research trends, collaboration networks, and the geographical distribution of expertise.

