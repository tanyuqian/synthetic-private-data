import fire
import json
import datasets
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from data_utils import get_topic_keywords


DATA_DIR = './pubmed'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOPIC_MODEL = './ctcl_topic'
GENERATOR_DIR = './ctcl_generator'


def main():
    dataset = datasets.load_dataset(
        'csv', data_files=f'{DATA_DIR}/train.csv', split='train').to_list()

    topic_model = BERTopic.load(
        TOPIC_MODEL, embedding_model=SentenceTransformer(EMBEDDING_MODEL))
    topic_keywords = get_topic_keywords(topic_model_dir=TOPIC_MODEL)

    topic_idxes, _ = topic_model.transform(
        documents=[example['text'] for example in dataset])

    with open(f'{DATA_DIR}/clustered_train.jsonl', 'w') as output_file:
        for example, topic_idx in zip(dataset, topic_idxes):
            topic_idx = int(topic_idx)
            print(json.dumps({
                **example,
                'topic': {
                    'idx': topic_idx,
                    'keywords': topic_keywords[topic_idx]
                }
            }), file=output_file)


if __name__ == '__main__':
    fire.Fire(main)