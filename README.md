**RussScholar-Seeker：A Python package for predicting whether a name is Russian**

<img width="410" alt="微信图片_20240409101854" src="https://github.com/TianciGao/RussScholar-Seeker/assets/153629778/ba73d175-091d-4439-925c-82cd9a97f7b9">

This project contains a series of programs designed to automatically identify and analyze Russian authors in academic papers. Utilizing the latest natural language processing technologies, it predicts the geographical attribute of names using a pre-trained BERT model to determine whether a given name is Russian.

## Principle

The core of the project is based on the `BertForSequenceClassification` model from the `transformers` library, trained with a specific dataset to distinguish Russian from non-Russian names. We first scrape metadata of academic papers, including titles and author names, from databases like DBLP. Then, we use this trained model to predict the names fetched, automatically identifying Russian authors.

## Production Process

1. **Data Preparation**: First, we collected a set of names labeled as Russian and non-Russian to serve as the dataset for training the model.
2. **Model Training**: We trained the model using `BertForSequenceClassification` and the collected dataset. During the training process, we adjusted the model parameters to achieve the best predictive performance.
3. **Data Scraping**: We wrote web scraping programs to fetch metadata of academic papers from databases like DBLP.
4. **Prediction and Analysis**: The fetched names were predicted using the trained model to identify Russian authors, and the related information was output.

## Usage Guide

Before using this tool, you need to install some necessary Python libraries, including `transformers`, `torch`, `requests`, and `beautifulsoup4`. The installation command is as follows:

```bash

pip install transformers torch requests beautifulsoup4

After that, you can run prediction.py to execute the Russian expert identification. The command might look like this:
python prediction.py

## Case Study: Identifying Russian Authors in AAAI 2021

One of the notable applications of this project was the analysis of academic papers from the AAAI 2021 conference, listed on DBLP. The goal was to identify papers with Russian authors, showcasing the model's ability to provide insights into geographical distributions of academic contributions.

### Results

The model successfully identified several papers with Russian authors, underlining the global collaboration in the field of Artificial Intelligence. Here are a few highlights from the analysis:

- **SMIL: Multimodal Learning with Severely Missing Modality**  
  Authors: Mengmeng Ma, Jian Ren, Long Zhao, Sergey Tulyakov, Cathy Wu, Xi Peng  
  Russian Author: Sergey Tulyakov

- **CHEF: Cross-modal Hierarchical Embeddings for Food Domain Retrieval**  
  Authors: Hai Xuan Pham, Ricardo Guerrero, Vladimir Pavlovic, Jiatong Li  
  Russian Author: Vladimir Pavlovic

- **Efficient Certification of Spatial Robustness**  
  Authors: Anian Ruoss, Maximilian Baader, Mislav Balunovic, Martin T. Vechev  
  Russian Authors: Mislav Balunovic, Martin T. Vechev

- **Adversarial Turing Patterns from Cellular Automata**  
  Authors: Nurislam Tursynbek, Ilya Vilkoviskiy, Maria Sindeeva, Ivan V. Oseledets  
  Russian Authors: Ilya Vilkoviskiy, Maria Sindeeva

These results not only demonstrate the practical utility of the Russian Expert Identifier in analyzing academic contributions but also highlight the diverse international collaboration within the AI research community.

### Implications

This case study underscores the potential of AI and NLP technologies in enhancing our understanding of academic landscapes. By automating the identification of geographical attributes of authors, we can gain valuable insights into global research trends, collaboration networks, and the geographical distribution of expertise.

