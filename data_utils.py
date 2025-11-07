from collections import Counter
import json
import numpy as np


def get_gen_dataset(topic_model_dir,
                    train_dataset,
                    hist_noise_multiplier,
                    n_gen_samples):
    topic_keywords = get_topic_keywords(topic_model_dir=topic_model_dir)

    cluster_keys = list(topic_keywords.keys())
    cluster_cnt = dict(
        Counter([example['topic']['idx'] for example in train_dataset]))

    dataset = []
    for cluster_key in cluster_keys:
        n_docs = cluster_cnt.get(cluster_key, 0)
        n_docs = n_docs + np.random.normal() * hist_noise_multiplier
        n_docs = max(0, int(round(n_docs)))
        for _ in range(n_docs):
          keywords = topic_keywords[cluster_key] if cluster_key != -1 else None

          example = {
              'topic': {'idx': cluster_key, 'keywords': keywords},
              'text': 'to_be_generated'
          }
          dataset.append(example)

    if len(dataset) < n_gen_samples:
        dataset = dataset * (n_gen_samples // len(dataset) + 1)
    else:
        dataset = dataset[:: len(dataset) // n_gen_samples]

    return dataset[: n_gen_samples]


def get_topic_keywords(topic_model_dir):
    topic_infos = json.load(open(f'{topic_model_dir}/topic_infos.json'))
    topic_keywords = {}
    for topic_info in topic_infos:
        topic_keywords[topic_info['topic_idx']] = topic_info['topic_words']

    return topic_keywords
