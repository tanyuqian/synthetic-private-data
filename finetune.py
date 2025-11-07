from functools import partial
import fire
import json
import math
import tqdm
import nltk
import jax
import jax.numpy as jnp
import optax
import datasets
from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM
from redco import Deployer, Trainer, Predictor

from data_utils import get_gen_dataset
from dp_utils import get_noise_multiplier, dp_train_step


DATA_DIR = './pubmed'
DOC_TYPE = 'medical paper abstract'
MODEL_DIR = './ctcl_generator'
TOPIC_MODEL_DIR = './ctcl_topic'
WORKDIR = './workdir'
JAX_SEED = 42
GLOBAL_BATCH_SIZE = 128
MIN_GLOBAL_STEPS = 1000
L2_NORM_CLIP = 1.0
LEARNING_RATE = 4e-4
LR_SCHEDULE_TYPE = 'linear'
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
DEFAULT_HIST_NOISE_MULTIPLIER = 10.
TOP_P = 0.95
N_GEN_SAMPLES = 400000
REPETITION_PENALTY = 1.5
MAX_SRC_LEN = 128
MAX_TGT_LEN = 512
MAX_WORD_LEN = 25
MAX_WORD_REPETITION = 20


def get_src(example, doc_type):
    conds = []
    conds.append(f'Document Type: {doc_type}')
    conds.append('Keywords: ' + ', '.join(example['topic']['keywords']))

    return '\n'.join(conds)


def collate_fn(examples, tokenizer, max_src_len, max_tgt_len, doc_type):
    model_inputs = tokenizer(
        [get_src(example, doc_type=doc_type) for example in examples],
        max_length=max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    decoder_inputs = tokenizer(
        [example['text'] for example in examples],
        max_length=max_tgt_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    model_inputs.update({
      'decoder_input_ids': decoder_inputs['input_ids'][:, :-1],
      'decoder_attention_mask': decoder_inputs['attention_mask'][:, :-1],
      'labels': decoder_inputs['input_ids'][:, 1:],
      'label_weights': decoder_inputs['attention_mask'][:, 1:],
    })
    return model_inputs


def loss_fn(rng, state, params, batch, is_training):
    labels, label_weights = batch.pop('labels'), batch.pop('label_weights')
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=rng, train=is_training).logits
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)
    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(rng, params, batch, model):
    return model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=rng).sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def filter_word_repetition(examples, max_word_len, max_word_repetition):
    results = []
    for example in tqdm.tqdm(examples, desc='Filtering'):
        word_cnt = {}
        overlong = False
        has_english = False
        for word in nltk.word_tokenize(example['pred']):
            word = word.strip().lower()
            if len(word) > max_word_len:
                overlong = True
            if word.isalpha():
                has_english = True
            if word not in ['the', 'a', ',', '.']:
                word_cnt[word] = word_cnt.get(word, 0) + 1
        if (has_english and
                len(word_cnt) > 0 and
                max(word_cnt.values()) <= max_word_repetition and
                not overlong):
            results.append(example)

    return results


def main(epsilon=4.0, per_device_batch_size=16, gen_per_device_batch_size=512):
    epsilon = float(epsilon)
    deployer = Deployer(workdir=WORKDIR, jax_seed=JAX_SEED)

    dataset = datasets.Dataset.from_json(
        path_or_paths=f'{DATA_DIR}/clustered_train.jsonl').to_list()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    with jax.default_device('cpu'):
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR, from_pt=True)
    model.generation_config.update(
        decoder_start_token_id=model.config.bos_token_id,
        max_length=MAX_TGT_LEN,
        no_repeat_ngram_size=0,
        num_beams=1,
        early_stopping=True,
        do_sample=True,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY)

    accumulate_grad_batches = deployer.get_accumulate_grad_batches(
        global_batch_size=GLOBAL_BATCH_SIZE,
        per_device_batch_size=per_device_batch_size)
    deployer.log_info(accumulate_grad_batches, title='accumulate_grad_batches')

    global_steps_per_epoch = len(dataset) // GLOBAL_BATCH_SIZE
    n_epochs = math.ceil(MIN_GLOBAL_STEPS / global_steps_per_epoch)
    deployer.log_info(n_epochs, title='n_epochs')

    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=len(dataset),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=LEARNING_RATE,
        schedule_type=LR_SCHEDULE_TYPE,
        warmup_ratio=WARMUP_RATIO)
    noise_multiplier, hist_noise_multiplier = get_noise_multiplier(
        epsilon=epsilon,
        default_hist_noise_multiplier=DEFAULT_HIST_NOISE_MULTIPLIER,
        has_hist=True,
        n_epochs=n_epochs,
        global_steps_per_epoch=global_steps_per_epoch,
        num_examples=len(dataset),
        global_batch_size=GLOBAL_BATCH_SIZE,
        workdir=deployer.workdir)

    # Scale noise to match effective noise with the desired global batch scaling
    # Ensure final noise ~ N(0, std² / global_batch_size²),
    # with global_batch_size = per_device_bs * device_count * accumulate_grad_batches
    noise_multiplier /= math.sqrt(accumulate_grad_batches) * jax.device_count()

    optimizer = optax.chain(
        optax.contrib.differentially_private_aggregate(
            l2_norm_clip=L2_NORM_CLIP,
            noise_multiplier=noise_multiplier,
            seed=JAX_SEED),
        optax.MultiSteps(
            optax.adamw(learning_rate=lr_schedule_fn, weight_decay=WEIGHT_DECAY),
            every_k_schedule=accumulate_grad_batches))

    collate_fn_kwargs = {
        'tokenizer': tokenizer,
        'max_src_len': MAX_SRC_LEN,
        'max_tgt_len': MAX_TGT_LEN,
        'doc_type': DOC_TYPE
    }
    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        apply_fn=model,
        loss_fn=loss_fn,
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        accumulate_grad_batches=accumulate_grad_batches,
        train_step_fn=partial(dp_train_step, l2_norm_clip=L2_NORM_CLIP))

    trainer.fit(
        train_examples=dataset,
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        pred_fn=partial(pred_fn, model=model),
        output_fn=partial(output_fn, tokenizer=tokenizer))

    gen_dataset = get_gen_dataset(
        topic_model_dir=TOPIC_MODEL_DIR,
        train_dataset=dataset,
        hist_noise_multiplier=hist_noise_multiplier,
        n_gen_samples=N_GEN_SAMPLES)
    preds = predictor.predict(
        examples=gen_dataset,
        per_device_batch_size=gen_per_device_batch_size,
        params=trainer.state.params,
        params_replicated=True)

    outputs = filter_word_repetition(
        [{'pred': pred} for pred in preds],
        max_word_repetition=MAX_WORD_REPETITION,
        max_word_len=MAX_WORD_LEN)
    if jax.process_index() == 0:
        output_filename = f'{deployer.workdir}/outputs_eps-{epsilon:.0f}.jsonl'
        with open(output_filename, 'w') as output_file:
            for output in outputs:
                output_file.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    fire.Fire(main)