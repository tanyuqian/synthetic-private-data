import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import l2_norm
import optax
import dp_accounting
from dp_accounting import dp_event
from dp_accounting import rdp


def get_noise_multiplier(epsilon,
                         default_hist_noise_multiplier,
                         has_hist,
                         n_epochs,
                         global_steps_per_epoch,
                         num_examples,
                         global_batch_size,
                         workdir):
    if epsilon == float('inf'):
        noise_multiplier = 0.0
        hist_noise_multiplier = 0.0
        actual_epsilon = float('inf')
    else:
        hist_noise_multiplier = default_hist_noise_multiplier
        noise_multiplier, actual_epsilon = (
            get_model_noise_multiplier_by_epsilon(
                has_hist=has_hist,
                hist_noise_multiplier=hist_noise_multiplier,
                epsilon=epsilon,
                steps=n_epochs * global_steps_per_epoch,
                num_examples=num_examples,
                batch_size=global_batch_size,
                target_delta=1.0 / (num_examples * np.log(num_examples))))

    if jax.process_index() == 0:
        noises = {
            'actual_epsilon': actual_epsilon,
            'model_noise_multiplier': noise_multiplier,
        }
        if has_hist:
            noises['hist_noise_multiplier'] = hist_noise_multiplier
        print(json.dumps(noises, indent=4), flush=True)

        noises_filename = f'{workdir}/noises_eps-{epsilon:.0f}.json'
        json.dump(noises, open(noises_filename, 'w'), indent=4)

    return noise_multiplier, hist_noise_multiplier


def get_model_noise_multiplier_by_epsilon(
    epsilon,
    steps,
    num_examples,
    batch_size,
    target_delta,
    has_hist,
    hist_noise_multiplier=10.0,
    low=0.0,
    high=100.0,
    n_attempts=30,
):
  def _compute_epsilon(noise_multiplier):
    gaussian_event = dp_accounting.GaussianDpEvent(
        noise_multiplier=noise_multiplier)
    sampled_event = dp_accounting.PoissonSampledDpEvent(
        sampling_probability=batch_size / num_examples, event=gaussian_event)
    training_event = dp_accounting.SelfComposedDpEvent(
        event=sampled_event, count=steps)

    accountant = dp_accounting.pld.PLDAccountant()
    if not has_hist:
      return accountant.compose(training_event).get_epsilon(target_delta)
    else:
      hist_event = dp_accounting.GaussianDpEvent(
          noise_multiplier=hist_noise_multiplier)
      composed_event = dp_accounting.ComposedDpEvent(
          [hist_event, training_event])
      return accountant.compose(composed_event).get_epsilon(target_delta)

  for _ in range(n_attempts):
    mid = (low + high) / 2.0
    if _compute_epsilon(noise_multiplier=mid) > epsilon:
      low = mid
    else:
      high = mid

  noise_multiplier = (low + high) / 2.0
  return noise_multiplier, _compute_epsilon(noise_multiplier=noise_multiplier)


# Copied from
# https://github.com/google/jax/blob/main/examples/differentially_private_sgd.py
def compute_epsilon(
        steps, num_examples, batch_size, noise_multiplier, target_delta):
    if num_examples * target_delta > 1.:
        print('Your delta might be too high.')
    q = batch_size / float(num_examples)
    orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            q, dp_event.GaussianDpEvent(noise_multiplier)), steps)
    return accountant.get_epsilon(target_delta)


# adapted from default_train_step(), added `loss_and_per_sample_grads`
# https://github.com/tanyuqian/redco/blob/master/redco/trainers/utils.py
def dp_train_step(rng,
                  state,
                  batch,
                  loss_fn,
                  lr_schedule_fn,
                  mesh,
                  compute_dtype,
                  l2_norm_clip):
    def loss_and_grads(rng_, batch_):
        loss, grads = jax.value_and_grad(
            lambda params: loss_fn(
                rng=rng_,
                state=state,
                params=params,
                batch=batch_,
                is_training=True)
        )(jax.tree.map(lambda x: x.astype(compute_dtype), state.params))

        # batch_ has only 1 sample
        grads_flat, grads_treedef = jax.tree.flatten(grads)
        grads_flat = jax.tree.map(lambda x: x[None], grads_flat)
        clipped, _ = optax.per_example_global_norm_clip(
            grads=grads_flat, l2_norm_clip=l2_norm_clip)
        return loss, jax.tree.unflatten(grads_treedef, clipped)

    def loss_and_clipped_per_sample_grads(rng_, batch_):
        batch_ = jax.tree.map(lambda x: x[:, None], batch_)
        loss, grads = jax.vmap(lambda b: loss_and_grads(rng_, b))(batch_)

        return loss.mean(), grads

    loss, grads = loss_and_clipped_per_sample_grads(rng, batch)
    grads = jax.lax.pmean(grads, axis_name='dp')
    new_state = state.apply_gradients(grads=jax.tree.map(
        lambda grad, param: grad.astype(param.dtype), grads, state.params))

    metrics = {'loss': loss, 'step': state.step, 'grad_norm': l2_norm(grads)}
    if lr_schedule_fn is not None:
        metrics['lr'] = lr_schedule_fn(state.step)
    if mesh is None:
        metrics = jax.lax.pmean(metrics, axis_name='dp')

    return new_state, metrics
