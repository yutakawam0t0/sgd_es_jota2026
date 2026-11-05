# Convergence Analysis of SGD under Expected Smoothness

Experimental code accompanying **Convergence Analysis of SGD under Expected Smoothness**, by **Yuta Kawamoto and Hideaki Iiduka**.

This repository studies SGD training and empirical expected smoothness (ES) diagnostics using ResNet-18 on CIFAR-100. It supports four learning-rate schedules and rolling-window estimation of nonnegative coefficients using nonnegative least squares (NNLS).

## Abstract

Stochastic gradient descent (SGD) is the workhorse of large-scale learning, yet classical analyses rely on assumptions that can be either too strong (bounded variance) or too coarse (uniform noise). The expected smoothness (ES) condition has emerged as a flexible alternative that ties the second moment of stochastic gradients to the objective value and the full gradient. This paper presents a self-contained convergence analysis of SGD under ES. We (i) refine ES with interpretations and sampling-dependent constants; (ii) derive bounds of the expectation of squared full gradient norm; and (iii) prove sublinear convergence rates with explicit residual errors for various step-size schedules. Furthermore, we empirically validate our theoretical framework by training a ResNet-18 model on the CIFAR-100 dataset. We uncover a two-phase dynamic in the gradient noise structure, transitioning from a suboptimality-dominated regime to a geometry-dominated regime. All proofs are given in full detail in the appendix. Our treatment unifies and extends recent theoretical threads in the literature.

## Source Files

The commands below use the repository filenames `main.py` and `utils.py`. If your copies are named `main_v1.py` and `utils_v1.py`, rename them to `main.py` and `utils.py`, respectively. The training script imports utilities from `utils`.

| File or directory | Purpose |
| :--- | :--- |
| `main.py` | Training, evaluation, learning-rate schedules, ES diagnostics, and W&B logging. |
| `utils.py` | Full-training-set gradient and loss evaluation, and sampling weights. |
| `models/` | Local model definitions; must export `ResNet18`. |
| `data/` | CIFAR-100 data, downloaded automatically. |

The local ResNet-18 implementation must support CIFAR-100 classification with 100 output classes.

## Dependencies

The supplied scripts import PyTorch, torchvision, NumPy, SciPy, and Weights & Biases (`wandb`). Install PyTorch and torchvision appropriate for your machine, along with:

```bash
python -m pip install numpy scipy wandb
```

The `wandb` package is imported even when logging is disabled. GPU execution uses `cuda:0` when CUDA is available; otherwise the script uses the CPU. Computing a full-training-set gradient at every epoch adds substantial work beyond ordinary training.

## Quick Start

Run the default configuration:

```bash
python main.py
```

This trains ResNet-18 on CIFAR-100 for **200 epochs**, using SGD without momentum, a batch size of **128**, a constant learning rate of **0.1**, and random seed **42**. W&B logging is disabled by default.

Display command-line options:

```bash
python main.py --help
```

### Weights & Biases

Authenticate before running with W&B logging:

```bash
wandb login
python main.py --use_wandb --optimizer sgd --sampling normal --scheduler constant
```

The project name is currently hard-coded as `SGD_ES_3`. Run names follow:

```text
{optimizer}_{scheduler}_{sampling}_s{seed}_w{window_size}
```

Change `wandb_project_name` or `wandb_exp_name` in `main.py` to customize these values.

On a Unix-like system, run in the background and capture both standard output and errors with:

```bash
nohup python -u main.py --use_wandb --optimizer sgd --sampling normal --scheduler cosine > log.log 2>&1 &
```

The script does not currently save model checkpoints, CSV metrics, or figure files. Use `--use_wandb` to retain the logged training and diagnostic metrics.

## Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--batch_size` | Integer | `128` | DataLoader batch size. Additional sampling weights are applied within each loaded batch. |
| `--optimizer` | String | `sgd` | `sgd`, `momentum`, `adam`, `rmsprop`, `adagrad`, `adamw`, or `amsgrad`. |
| `--sampling` | String | `normal` | `normal`, `replacement`, `independent`, or `tau-nice`; see implementation details below. |
| `--use_wandb` | Flag | Disabled | Enable W&B logging. Pass the flag without a value. |
| `--seed` | Integer | `42` | Seed for Python, NumPy, and PyTorch random number generators. |
| `--window_size` | Integer | `50` | Maximum number of epoch snapshots retained for NNLS fitting. Use at least `5` to obtain coefficient estimates. |
| `--lr` | Float | `0.1` | Initial learning rate for `sgd` and `momentum`. Other optimizers use hard-coded initial learning rates. |
| `--scheduler` | String | `constant` | `constant`, `harmonic`, `polynomial`, or `cosine`. |
| `--poly_alpha` | Float | `0.51` | Polynomial-decay exponent. Use an exponent in `(0.5, 1)` for the corresponding schedule analyzed in the paper. |

Model architecture, dataset, training duration, and device selection are set inside the script. There are no `--model`, `--epochs`, `--dataset`, or `--device` arguments. This version uses command-line arguments and does not load JSON configuration files.

For `momentum`, the code uses momentum `0.9` and weight decay `5e-4`. Plain `sgd` uses momentum `0.0` and no weight decay. The initial learning rates for `adam`, `adamw`, and `amsgrad` are `0.001`; those for `rmsprop` and `adagrad` are `0.01`. The selected scheduler also applies to these optimizers. The paper's SGD analysis should not be interpreted as a convergence guarantee for every optimizer exposed by the script.

## Learning-Rate Schedules

Schedules are updated **once per epoch**, after training, testing, and ES diagnostics. With zero-based epoch index $e$, initial learning rate $\eta_0$, and $E=200$ epochs:

| Schedule | Learning rate |
| :--- | :--- |
| `constant` | $\eta_e=\eta_0$ |
| `harmonic` | $\eta_e=\eta_0/(e+1)$ |
| `polynomial` | $\eta_e=\eta_0/(e+1)^{\alpha}$ |
| `cosine` | $\eta_e=\eta_0[1+\cos(\pi e/E)]/2$ |

These are epoch-based implementations; the theoretical analysis indexes individual SGD iterations.

```bash
# Constant learning rate
python main.py --use_wandb --optimizer sgd --scheduler constant --lr 0.1

# Harmonic decay
python main.py --use_wandb --optimizer sgd --scheduler harmonic --lr 0.1

# Polynomial decay
python main.py --use_wandb --optimizer sgd --scheduler polynomial --lr 0.1 --poly_alpha 0.51

# Cosine annealing
python main.py --use_wandb --optimizer sgd --scheduler cosine --lr 0.1
```

## Sampling Implementation

The DataLoader first produces a shuffled batch. The script then constructs weights within that batch and differentiates the mean weighted per-example loss. Thus, `--batch_size` specifies the outer batch size, rather than the number of subsequent draws or selected examples.

| Option | Current behavior within a loaded batch |
| :--- | :--- |
| `normal` | Assign weight one to every example. |
| `replacement` | Draw 32 indices uniformly with replacement and use inverse-probability count weights. |
| `independent` | Include each example independently with probability $64/m$, where $m$ is the current batch size; use inverse-probability weights. |
| `tau-nice` | Currently calls the same independent Bernoulli sampler as `independent`. |

**Implementation limitation:** the current `tau-nice` option does not select exactly 64 distinct examples and therefore does not implement the fixed-cardinality $\tau$-nice sampling analyzed in the paper. It must be corrected before using this option to reproduce a theoretical $\tau$-nice comparison.

The values 32 and 64 are hard-coded. For `independent` and the current `tau-nice` implementation, every loaded batch, including the final partial batch, must contain at least 64 examples so that $64/m\leq1$. The default batch size of 128 satisfies this requirement for CIFAR-100's 50,000 training examples.

## Empirical ES Diagnostics

After each epoch, the script performs the following calculations:

1. Compute the average loss and full parameter gradient over the training set without data augmentation, using evaluation mode.
2. Estimate the stochastic-gradient second moment by averaging squared gradient norms over **50 training batches**, using training mode and the selected sampling weights.
3. Retain up to `--window_size` epoch snapshots and, once at least five snapshots are available, fit nonnegative coefficients using `scipy.optimize.nnls`.

For snapshots indexed by $j$, the fitted relationship is

$$
\widehat{\mathbb{E}\|g_j\|^2}
\approx 2\widehat A f_j
+\widehat B\|\nabla f_j\|^2+\widehat C.
$$

The design matrix contains the columns $2f_j$, $\|\nabla f_j\|^2$, and $1$. The current implementation uses the **raw loss** $f_j$, does not subtract an estimated optimum, and does not normalize the design-matrix columns before NNLS. The 50-batch estimate uses an arithmetic mean, not a median.

The fitted coefficients are empirical diagnostics. NNLS minimizes a fitting error subject to coefficient nonnegativity; it does not enforce the ES upper-bound inequality or establish globally valid ES constants. In addition, the full gradient is evaluated without augmentation in evaluation mode, whereas sampled gradients use augmented data in training mode. The two measurements therefore use different evaluation settings. Diagnostic training-mode forward passes can also update BatchNorm running statistics.

### Logged Metrics

| W&B key | Description |
| :--- | :--- |
| `epoch` | Zero-based epoch index. |
| `lr` | Learning rate used for the current epoch. |
| `training_loss` | Mean of the weighted training batch losses. |
| `training_acc` | Training accuracy, in percent. |
| `test_loss` | Mean of the test batch losses. |
| `test_acc` | Test accuracy, in percent. |
| `avg_grad_norm` | Mean training mini-batch gradient norm; this is not a squared norm. |
| `f_val` | Full-training-set average loss without augmentation, evaluated in evaluation mode. |
| `grad_full_norm2` | Squared norm of the corresponding full-training-set gradient. |
| `ES_LHS` | Mean squared sampled-gradient norm over 50 diagnostic batches. |
| `A`, `B`, `C` | Nonnegative NNLS coefficient estimates, available after at least five snapshots. |
| `ES_RHS` | Fitted value for the most recent snapshot. |
| `residual` | `ES_LHS - ES_RHS` for the most recent snapshot. |

## Reproducibility

The script seeds Python, NumPy, and PyTorch, enables deterministic cuDNN behavior, and disables cuDNN benchmarking. Record the source revision, command-line arguments, package versions, and hardware alongside each run. Identical seeds alone do not guarantee bitwise-identical results across platforms or software versions.

For example:

```bash
python main.py --use_wandb --optimizer sgd --sampling normal --scheduler polynomial --poly_alpha 0.51 --batch_size 128 --lr 0.1 --window_size 50 --seed 42
```

Keep the diagnostic procedure and its frequency fixed when comparing runs, because diagnostics consume random samples and may update BatchNorm statistics.
