from functools import partial
import fire
import json
import math
import numpy as np
import jax.numpy as jnp
import optax
import datasets
from redco import Deployer, Trainer, Predictor
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM


DATA_DIR = './pubmed'
JAX_SEED = 42
DOWNSTREAM_MODEL_NAME = 'prajjwal1/bert-mini'
TRAIN_SIZE = 400000
MAX_LENGTH = 512
LEARNING_RATE = 3e-4
GLOBAL_BATCH_SIZE = 64
N_EPOCHS = 1
LR_SCHEDULE_TYPE = 'linear'
WARMUP_RATIO = 0.
WEIGHT_DECAY = 0.01


def collate_fn(examples, tokenizer, max_length):
    batch = tokenizer(
        [example['text'] for example in examples],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np')
    labels = np.full_like(batch['input_ids'], tokenizer.pad_token_id)
    label_weights = np.zeros_like(batch['input_ids'])
    labels[:, :-1] = batch['input_ids'][:, 1:]
    label_weights[:, :-1] = batch['attention_mask'][:, 1:]
    batch.update({'labels': labels, 'label_weights': label_weights})

    return batch


def loss_fn(rng, state, params, batch, is_training):
    labels = batch.pop('labels')
    label_weights = batch.pop('label_weights')
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=rng, train=is_training)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(rng, batch, params, model):
    labels = batch.pop('labels')
    label_weights = batch.pop('label_weights')
    logits = model(**batch, params=params, train=False)[0]

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)
    loss = jnp.sum(loss * label_weights, axis=-1) / jnp.clip(
        jnp.sum(label_weights, axis=-1), min=1e-5)
    label_pred = jnp.argmax(logits, axis=-1)

    n_token_match = jnp.sum((label_pred == labels) * label_weights, axis=-1)
    n_target_tokens = jnp.sum(label_weights, axis=-1)

    return {
        'loss': loss,
        'n_token_match': n_token_match,
        'n_target_tokens': n_target_tokens
    }


def main(syn_data_file='workdir/outputs_eps-4.jsonl', per_device_batch_size=8):
    deployer = Deployer(workdir=None, jax_seed=JAX_SEED)

    tokenizer = AutoTokenizer.from_pretrained(DOWNSTREAM_MODEL_NAME)
    model = FlaxAutoModelForCausalLM.from_pretrained(
        DOWNSTREAM_MODEL_NAME, from_pt=True)
    if not model.config.is_decoder:
        model.config.is_decoder = True

    synthetic_examples = \
        [{'text': json.loads(line)['pred']} for line in open(syn_data_file)]
    synthetic_examples *= (TRAIN_SIZE // len(synthetic_examples) + 1)
    synthetic_examples = synthetic_examples[:TRAIN_SIZE]

    private_examples = datasets.Dataset.from_csv(
        path_or_paths=f'{DATA_DIR}/test.csv').to_list()

    accumulate_grad_batches = deployer.get_accumulate_grad_batches(
        global_batch_size=GLOBAL_BATCH_SIZE,
        per_device_batch_size=per_device_batch_size)
    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=len(synthetic_examples),
        per_device_batch_size=per_device_batch_size,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        schedule_type=LR_SCHEDULE_TYPE,
        warmup_ratio=WARMUP_RATIO)

    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=lr_schedule_fn, weight_decay=WEIGHT_DECAY),
        every_k_schedule=accumulate_grad_batches)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn, tokenizer=tokenizer, max_length=MAX_LENGTH),
        apply_fn=model,
        loss_fn=loss_fn,
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        accumulate_grad_batches=accumulate_grad_batches)
    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn, tokenizer=tokenizer, max_length=MAX_LENGTH),
        pred_fn=partial(pred_fn, model=model))

    trainer.fit(
        train_examples=synthetic_examples,
        per_device_batch_size=per_device_batch_size,
        n_epochs=N_EPOCHS)
    preds = predictor.predict(
        examples=private_examples,
        per_device_batch_size=per_device_batch_size,
        params=trainer.state.params,
        params_replicated=True)

    loss = np.mean([pred['loss'] for pred in preds])
    token_acc = np.sum([pred['n_token_match'] for pred in preds]) / np.sum(
        [pred['n_target_tokens'] for pred in preds])
    print(json.dumps({
        'loss': loss.item(),
        'perplexity': math.exp(loss.item()),
        'token_accuracy': token_acc.item(),
    }, indent=4))


if __name__ == '__main__':
    fire.Fire(main)
