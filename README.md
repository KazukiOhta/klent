# Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games

This repository provides supplementary code for the ICML 2026 paper
[**"Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games."**](https://arxiv.org/abs/2602.10894)

The code contains a compact implementation of our algorithm KLENT for training and evaluating agents in board game environments.

## Installation

Please run the following commands to install the dependencies.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execution

Please run the following command to train and evaluate an agent in a 9x9 Go environment without Weights & Biases logging.

```
python3 main.py env_id=go_9x9 wandb_on=false
```

Choose an environment from the following list: `connect_four`, `animal_shogi`, `gardner_chess`, `go_9x9`, `hex`, and `othello`. If you do not specify, `connect_four` will be chosen by default.

To enable Weights & Biases logging, set `wandb_on=true`.

Please make sure that you have activated the virtual environment before the execution. The execution may require a GPU with CUDA 12.

## Contents
* `main.py` is the main file that trains and evaluates the agent.
* `resnet.py` provides the residual network architecture with policy and action-value heads.
* `util.py` provides a useful function for handling `jax.random.key`.
* `README.md` is this file.
* `requirements.txt` provides the information of dependencies. 
