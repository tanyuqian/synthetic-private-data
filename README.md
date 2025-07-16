# Synthesizing Privacy-Preserving Text Data


This repo is for the following paper:

[Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs](https://arxiv.org/abs/2503.12347) \
Bowen Tan, Zheng Xu, Eric Xing, Zhiting Hu, Shanshan Wu \
ICML 2025

## Getting Started:
* Synthetic data offers a promising path to train models while preserving data privacy. 
* We propose CTCL (Data Synthesis with ConTrollability and CLustering), a framework for generating privacy-preserving synthetic data *without extensive prompt engineering or billion-scale LLM finetuning.* 


## Pretrained CTCL-Topic and CTCL-Generator
We release the pretrained topic model and generator via [this GDrive link](https://drive.google.com/file/d/1sbda6ROyMewThuoDA3bxP71ucihcf7qJ/view?usp=drive_link).

They can also be downloaded by `gdown`:
```
pip install gdown

gdown 1sbda6ROyMewThuoDA3bxP71ucihcf7qJ
unzip ctcl_pretrained.zip
```

### CTCL-Topic
The topic model implementation is based on `bertopic`. Below is a demo code.

```python
import fire
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOPIC_MODEL = './ctcl_topic'
GENERATOR_DIR = './ctcl_generator'

DOCS = [
    "Ammon News : Lower House Speaker Abdel Karim Doghmi Tuesday commended the Audit Bureau's oversight role to fight corruption and protect public money, and pledged to give utmost attention to the bureau's reports.\nHe said as he welcomed Audit Bureau President Assem Haddad, who handed him the annual report, that the rule of law is a priority for His Majesty King Abdullah that was underlined in the Speech from the Throne opening Parliament's ordinary session.\nThe Audit Bureau chief also handed a copy of the report to Senate President Faisal Fayez, who hailed the agency for adopting world-class oversight practices to preserve public funds and correct the imbalances in the government agencies under its jurisdiction.\nFayez stressed cooperation between the Senate and the bureau, which, he said, is worthy of more funding and technical support to carry out its tasks.\nLower House Speaker Abdel Karim Doghmi Tuesday commended the Audit Bureau's oversight role to fight corruption and protect public money, and pledged to give utmost attention to the bureau's reports.\nLower House, Senate heads underline Audit Bureau oversight role",
    "The Institute for Voluntary Action Research (IVAR) is working in partnership with London Funders, the Association of Charitable Foundations and a number of independent funders on research to explore place-based funding approaches.\nThe term place-based funding is used to describe a spectrum of approaches. At one end of the spectrum, it may be used simply to refer to grant-making limited by geography \u2013 a decision to fund only in specified geographic areas; at the other it may refer to long-term and multi-faceted collaborative partnerships aiming to achieve significant change. In most cases, it is more than just a term to describe the target location of funding; it also describes a style and philosophy of funding.\nDevelop a series of case studies that focus on learning about the approaches in terms of rationale, success and failure.\nThe paper draws on current practice and literature to explore what place-based funding is, looking at: how it is delivered; the different types of approaches and roles funders play; benefits and challenges of working with place.\nThis briefing paper is a working document that will be refreshed during the course of the research. Readers who would like to give feedback on the paper or who are interested in the study are invited to contact Eliza Buckley."
]

def main():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL).to('cuda')
    topic_model = BERTopic.load(TOPIC_MODEL, embedding_model=embedding_model)

    embeddings = embedding_model.encode(DOCS, show_progress_bar=True)
    topic_idxes, _ = topic_model.transform(
        documents=DOCS, embeddings=embeddings)

    topic_keywords = {
        topic_idx: keywords
        for topic_idx, keywords in zip(
            list(topic_model.get_topic_info()['Topic']),
            list(topic_model.get_topic_info()['Representation']))
        if topic_idx >= 0
    }

    print(json.dumps([
        {'document': DOCS[0], 'keywords of topic': topic_keywords[topic_idxes[0]]},
        {'document': DOCS[1], 'keywords of topic': topic_keywords[topic_idxes[1]]}
    ], indent=4))


if __name__ == '__main__':
    fire.Fire(main)
```

### CTCL-Generator
We release the pretrained generator model in HuggingFace `transformers` format. Below is a demo code.

```python
import fire
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GENERATOR_DIR = './ctcl_generator'
CONDS = [
    "Communication Style:  Professional and polished.\nKeywords: KeyWords: Leasing, commercial/residential, maintenance, residential investment",
    "Document Type: Football Transfer Update\nKeywords: Manchester City, Douglas Costa, Juventus, transfer, summer window, Premier League,  Serie A"
]

def main():
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_DIR).to('cuda')

    inputs = tokenizer(
        CONDS, max_length=64, truncation=True, padding=True, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_length=256)

    docs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(json.dumps([
        {'condition': CONDS[0], 'generation': docs[0]},
        {'condition': CONDS[1], 'generation': docs[1]}
    ], indent=4))


if __name__ == '__main__':
    fire.Fire(main)
```

*More code (for pretraining, DP fine-tuning, evaluation, etc.) coming soon...*