import os
os.environ["KERAS_BACKEND"] = "jax"

from datetime import datetime
from time import time
import itertools

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pgx
import pytz
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel
from pgx.experimental import auto_reset
from typing import get_args

from resnet import PQNet
from util import KeyGenerator



class Config(BaseModel):
    env_id: str = "connect_four"
    seed: int = 0
    # KLENT hyper parameters
    selfplay_vmap: int = 1024
    selfplay_step: int = 2048
    alpha: float = 0.03
    beta: float = 0.1
    tau: float = 8.0
    fitting_batch_size: int = 4096
    fitting_epochs: int = 1
    limit_simulator_evaluations: int = 10**9
    # ResNet hyper parameters
    num_channels: int = 128
    num_blocks: int = 6
    num_params: int = 0
    zero_init: bool = True
    # Optimizer hyper parameters
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    # Server information
    host_name: str = os.uname().nodename
    device_kind:str = jax.local_devices()[0].device_kind
    # Evaluation
    evaluation_vmap: int = 1024
    # Checkpoints
    save_interval: int = 50
    checkpoints_dir: str = "./checkpoints"
    # Weights and Biases
    wandb_on: bool = True
    comment: str = ""

def network(observations, params):
    observations = jax.vmap(boolify)(observations)
    outputs, _ = model.stateless_call(**params, inputs=observations, training=False)
    return outputs["logits"], outputs["qvalue"]

def encode(observation):
    return jnp.packbits(observation.astype(jnp.bool).flatten())

def decode(code):
    return jnp.unpackbits(code)[:prod_observation_shape].reshape(env.observation_shape).astype(jnp.float32)

def boolify(observation):
    return decode(encode(observation))

def calculate_targets(R, V, T):
    V_next = jnp.concatenate([V[1:], jnp.array([jnp.nan])])
    RVT = jnp.stack([R, V_next, T], axis=1)
    lambda_ = jnp.exp(-1/jnp.clip(config.tau, min=1e-12))
    gamma = -1

    def body_fn(carry, rvt):
        r, v, terminated = rvt
        target = r + gamma*((1-lambda_)*v + lambda_*carry)
        return jax.lax.cond((terminated>0).all(), lambda: (r, r), lambda: (target, target))

    _, G = jax.lax.scan(body_fn, jnp.nan, RVT, reverse=True)
    return G

@jax.jit
def selfplay(rng_key, params):
    n_vmap = config.selfplay_vmap
    n_step = config.selfplay_step
    step_fn = auto_reset(env.step, env.init)

    def body_fn(i_step, loop_state):
        states, C, A, R, T, P, V, STATS, rng_key = loop_state
        rng_key, key1, key2 = jax.random.split(rng_key, 3)

        # Calculate improved policy
        a, b = config.alpha, config.beta
        logits, qvalue = network(states.observation, params)
        improved_logits = (b * logits + qvalue) / (a + b + 1e-12)
        improved_logits = improved_logits + jnp.log(states.legal_action_mask)
        improved_policies = jax.nn.softmax(improved_logits, axis=-1)

        # Choose action
        actions = jax.vmap(lambda p, key: jax.random.choice(key, a=env.num_actions, p=p))(improved_policies, jax.random.split(key1, n_vmap))

        # Store sample
        C = C.at[i_step].set(jax.vmap(encode)(states.observation))
        A = A.at[i_step].set(actions)
        P = P.at[i_step].set(improved_policies)
        V = V.at[i_step].set(jnp.sum(improved_policies*qvalue, axis=1))
        current_player = states.current_player
        states = jax.vmap(step_fn)(states, actions, jax.random.split(key2, n_vmap))
        R = R.at[i_step].set(jax.vmap(lambda R, c: R[c])(states.rewards, current_player))
        T = T.at[i_step].set(states.terminated)

        # Calculate stats
        prior_policies = jax.nn.softmax(logits, axis=-1)
        STATS = STATS.at[i_step].set(jnp.array([
            jnp.sum(prior_policies   *qvalue, axis=1),                  # return_0
            jnp.sum(improved_policies*qvalue, axis=1),                  # return_1
            jax.vmap(kl_divergence)(prior_policies   , prior_policies), # kl_0
            jax.vmap(kl_divergence)(improved_policies, prior_policies), # kl_1
            jax.vmap(entropy)(prior_policies   ),                       # ent_0
            jax.vmap(entropy)(improved_policies),                       # ent_1
        ]).T)

        return states, C, A, R, T, P, V, STATS, rng_key

    code = encode(jnp.zeros(env.observation_shape))
    C = jnp.zeros((n_step, n_vmap, *code.shape), dtype=code.dtype)
    A = jnp.zeros((n_step, n_vmap), dtype=jnp.int32)
    R = jnp.zeros((n_step, n_vmap))
    T = jnp.zeros((n_step, n_vmap), dtype=jnp.bool_)
    P = jnp.zeros((n_step, n_vmap, env.num_actions))
    V = jnp.zeros((n_step, n_vmap))
    STATS = jnp.zeros((n_step, n_vmap, 6))
    key1, key2 = jax.random.split(rng_key)
    states = jax.vmap(env.init)(jax.random.split(key1, n_vmap))
    _, C, A, R, T, P, V, STATS, _ = jax.lax.fori_loop(0, n_step, body_fn, (states, C, A, R, T, P, V, STATS, key2))

    # Transpose into (n_vmap, n_step, ...).
    C, A, R, T, P, V = map(lambda X: jnp.swapaxes(X, 0, 1), (C, A, R, T, P, V))

    # Calculate the value target
    G = jax.vmap(calculate_targets)(R, V, T)

    C = C.reshape((n_vmap*n_step, *C.shape[2:]))
    A = A.reshape((n_vmap*n_step, 1))
    P = P.reshape((n_vmap*n_step, env.num_actions))
    G = G.reshape((n_vmap*n_step, 1))

    # Aggregate stats
    STATS = jnp.mean(STATS, axis=(0, 1))

    return C, A, P, G, STATS

@jax.jit
def evaluate(rng_key, params, opp_coef):
    our_player = 0
    rng_key, sub_key = jax.random.split(rng_key)
    n_vmap = config.evaluation_vmap
    states = jax.vmap(env.init)(jax.random.split(sub_key, n_vmap))

    def body_fn(loop_state):
        rng_key, states, rewards = loop_state
        rng_key, sub_key = jax.random.split(rng_key)
        logits, qvalue = network(states.observation, params)
        our_logits = 10000 * logits
        opp_logits = opp_coef * baseline(states.observation)[0]
        logits = jnp.where((states.current_player==our_player).reshape(-1, 1), our_logits, opp_logits)
        logits = logits + jnp.log(states.legal_action_mask)
        actions = jax.random.categorical(sub_key, logits, axis=-1)
        states = jax.vmap(env.step)(states, actions)
        rewards = rewards + states.rewards[jnp.arange(n_vmap), our_player]
        return rng_key, states, rewards

    _, _, reward = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (rng_key, states, jnp.zeros(n_vmap))
    )

    W, D, L = jnp.mean(reward==1), jnp.mean(reward==0), jnp.mean(reward==-1)
    return W, D, L

def enrich_log(log):
    iteration = log["cost/iteration"]
    sim_plan = 0
    sim_play = config.selfplay_vmap * config.selfplay_step * iteration

    log["cost/simulator_evaluations/planning"] = sim_plan
    log["cost/simulator_evaluations/playing"] = sim_play
    log["cost/simulator_evaluations/total"] = sim_plan + sim_play
    log["cost/simulator_evaluations/total [million]"] = (sim_plan + sim_play) / (10**6)

    log["cost/hours/total"] = log["cost/hours/selfplay"] + log["cost/hours/preprocess"] + log["cost/hours/fit"]
    for key in ["selfplay", "fit", "total"]:
        log[f"cost/gpu_hours/{key}"] = log[f"cost/hours/{key}"]

    for opp in list(map(lambda key: key.split("/")[1], filter(lambda key: "vs_baseline" in key and "win_rate" in key, log.keys()))):
        W, D, L = log[f"eval/{opp}/win_rate"], log[f"eval/{opp}/draw_rate"], log[f"eval/{opp}/lose_rate"]
        log[f"eval/{opp}/avg_R"] = 1*W +   0*D +(-1)*L
        log[f"eval/{opp}/score"] = 1*W + 0.5*D +   0*L
        log[f"score/{opp}"]      = 1*W + 0.5*D +   0*L

    log["train/total_loss"] = log["train/policy_loss"] + log["train/qvalue_loss"]

    log["stats/sample_util_ratio"] = log["cost/frames/used"] / log["cost/frames/total"]
    log["stats/effective_actions"] = log["stats/policy_target_mean_exp_entropy"]
    log["stats/effective_actions_v2"] = float(np.exp(log["stats/policy_target_mean_entropy"]))

    for s in ["return", "kl", "ent"]:
        log[f"selfplay_stats/{s}_diff"] = log[f"selfplay_stats/{s}_1"] - log[f"selfplay_stats/{s}_0"]

    return dict(sorted(log.items()))

def get_params(model):
    return {
        "trainable_variables": tuple(jnp.array(var.numpy()) for var in model.trainable_variables),
        "non_trainable_variables": tuple(jnp.array(var.numpy()) for var in model.non_trainable_variables),
    }

def qvalue_loss_fn(y_true, y_pred):
    A = y_true[:, 0].astype(int)
    G = y_true[:, 1]
    q_pred = y_pred[jnp.arange(y_pred.shape[0]), A]
    squared_error = jnp.square(q_pred - G)
    return squared_error

def entropy(p):
    return jnp.sum(jnp.where(p == 0, 0, - p * jnp.log(p)))

def kl_divergence(p, q):
    return jnp.sum(jnp.where(p == 0, 0, p * (jnp.log(p) - jnp.log(q))))

# Config setup
conf_dict = OmegaConf.from_cli()
config = Config(**conf_dict)
env = pgx.make(config.env_id)
prod_observation_shape = int(jnp.prod(jnp.array(env.observation_shape)))

# Keras setup
model = PQNet(
    input_shape=env.observation_shape,
    num_actions=env.num_actions,
    zero_init=config.zero_init,
    num_channels=config.num_channels,
    num_blocks=config.num_blocks
)

losses = {
    "logits": keras.losses.CategoricalCrossentropy(from_logits=True),
    "qvalue": qvalue_loss_fn,
}
model.compile(
    optimizer=getattr(keras.optimizers, config.optimizer)(learning_rate=config.learning_rate),
    loss = losses,
    metrics = losses,
)
config.num_params = model.count_params()

# Baseline setup
baseline_id = config.env_id + "_v0"
if baseline_id in get_args(pgx.BaselineModelId):
    baseline = pgx.make_baseline_model(baseline_id)
else:
    def baseline(obs):
        return jnp.zeros((obs.shape[0], env.num_actions)), None

# Checkpoints setup
ckpt_dir = os.path.join(config.checkpoints_dir, f"{config.env_id}_{config.seed}")
os.makedirs(ckpt_dir, exist_ok=True)

jit_vmap_decoder = jax.jit(jax.vmap(decode))
def data_generator(C, A, P, G, rng_key):
    N = len(C)
    batch_size = config.fitting_batch_size
    while True:
        rng_key, sub_key = jax.random.split(rng_key)
        idx = jax.random.permutation(sub_key, jnp.arange(N))
        for start in range(0, N, batch_size):
            c, a, p, g = (x[idx[start:start+batch_size]] for x in (C, A, P, G))
            o = jit_vmap_decoder(c)
            target = {"logits": p, "qvalue": jnp.concatenate([a, g], axis=1)}
            yield o, target

def main():
    if config.wandb_on:
        wandb.init(project="klent", config=config.model_dump())
    key = KeyGenerator(config.seed)

    hours_selfplay, hours_preprocess, hours_fit = 0.0, 0.0, 0.0
    frames_total, frames_used = 0, 0

    for iteration in itertools.count():
        print(f"Iteration: {iteration}")
        t0 = time()
        selfplay_output = selfplay(key(), get_params(model))
        t1 = time()
        C, A, P, G, STATS = jax.device_get(selfplay_output)
        del selfplay_output
        jax.clear_caches()
        mask = jnp.squeeze(jnp.isfinite(G))
        C, A, P, G = map(lambda x: x[mask], (C, A, P, G))
        N = len(C)
        t2 = time()
        history = model.fit(
            x = data_generator(C, A, P, G, key()),
            steps_per_epoch = N//config.fitting_batch_size,
            epochs=config.fitting_epochs,
        )
        t3 = time()

        eval_log = {}
        for opp_coef in [1.00]:
            W, D, L = evaluate(key(), get_params(model), opp_coef)
            eval_log[f"eval/vs_baseline_{int(100*opp_coef):03}/win_rate"] = float(W)
            eval_log[f"eval/vs_baseline_{int(100*opp_coef):03}/draw_rate"] = float(D)
            eval_log[f"eval/vs_baseline_{int(100*opp_coef):03}/lose_rate"] = float(L)


        # logging
        hours_selfplay += (t1-t0)/3600
        hours_preprocess += (t2-t1)/3600
        hours_fit += (t3-t2)/3600
        frames_total += config.selfplay_vmap*config.selfplay_step
        frames_used += N

        ENT = np.array(jax.lax.map(entropy, P))
        log = enrich_log(eval_log | {
            "cost/iteration": iteration + 1,
            "cost/hours/selfplay": hours_selfplay,
            "cost/hours/preprocess": hours_preprocess,
            "cost/hours/fit": hours_fit,
            "cost/frames/total": frames_total,
            "cost/frames/used": frames_used,
            "train/policy_loss": float(np.mean(history.history["logits_categorical_crossentropy"])),
            "train/qvalue_loss": float(np.mean(history.history["qvalue_qvalue_loss_fn"])),
            "stats/policy_target_mean_entropy": float(np.mean(ENT)),
            "stats/policy_target_mean_exp_entropy": float(np.mean(np.exp(ENT))),
            "selfplay_stats/return_0": float(STATS[0]),
            "selfplay_stats/return_1": float(STATS[1]),
            "selfplay_stats/kl_0": float(STATS[2]),
            "selfplay_stats/kl_1": float(STATS[3]),
            "selfplay_stats/ent_0": float(STATS[4]),
            "selfplay_stats/ent_1": float(STATS[5]),
        })
        print(log)
        if config.wandb_on:
            wandb.log(log)
        del C, A, P, G, ENT, STATS, W, D, L, history
        jax.clear_caches()

        if (iteration+1)%config.save_interval==0:
            model.save(ckpt_dir+f"/{iteration+1:04}.keras")
        if log["cost/simulator_evaluations/total"] >= config.limit_simulator_evaluations:
            model.save(ckpt_dir+"/final.keras")
            break

print(config)
result = main()
