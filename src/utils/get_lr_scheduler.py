import torch.optim as optim
import math
from .get_config_value import get_config_value


def diminishing_lr_lambda(steps):
    return 1 / math.sqrt(steps + 1)


def linear_lr_lambda(steps, total_steps):
    return 1 - (steps / total_steps)


def exp_growth_lr_lambda(epoch, eta_init, incr_interval, epochs, eta_max=None, lr_growth_rate=None):
    if lr_growth_rate is not None:
        if eta_max is not None:
            exponent = (epochs - incr_interval) / incr_interval
            a = (eta_max - eta_init) / (lr_growth_rate ** exponent - 1)
            b = eta_init - a
            return min((1 / eta_init) * (a * (lr_growth_rate ** (epoch // incr_interval)) + b), (eta_max / eta_init))
        else:
            return lr_growth_rate ** (epoch // incr_interval)

    elif eta_max is not None:
        gamma = (eta_max / eta_init) ** (incr_interval / (epochs - incr_interval))
        return min(gamma ** (epoch // incr_interval), (eta_max / eta_init))

    else:
        raise ValueError("Either 'epochs' and 'eta_max' or 'growth_rate' must be provided.")


def exp_warmup_const_lr_lambda(epoch, warmup_epochs, warmup_interval, eta_init, eta_max=None, lr_growth_rate=None):
    if lr_growth_rate is not None:
        if epoch < warmup_epochs:
            return lr_growth_rate ** (epoch // warmup_interval)
        else:
            return lr_growth_rate ** ((warmup_epochs - warmup_interval) // warmup_interval)

    elif eta_max is not None:
        if epoch < warmup_epochs:
            gamma = (eta_max / eta_init) ** (warmup_interval / (warmup_epochs - warmup_interval))
            return min(gamma ** (epoch // warmup_interval), (eta_max / eta_init))
        else:
            return (eta_max / eta_init)

    else:
        raise ValueError("Either 'eta_max' or 'growth_rate' must be provided.")


def exp_warmup_cosine_lr_lambda(epoch, warmup_epochs, warmup_interval, epochs, eta_init, eta_min=0, eta_max=None, lr_growth_rate=None):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    if lr_growth_rate is not None:
        if epoch < warmup_epochs:
            return lr_growth_rate ** (epoch // warmup_interval)
        else:
            eta_max = lr_growth_rate ** ((warmup_epochs - warmup_interval) // warmup_interval)
            eta_min = eta_min / eta_init
            return eta_min + (eta_max - eta_min) * cosine_decay

    elif eta_max is not None:
        if epoch < warmup_epochs:
            gamma = (eta_max / eta_init) ** (warmup_interval / (warmup_epochs - warmup_interval))
            return min(gamma ** (epoch // warmup_interval), (eta_max / eta_init))
        else:
            return (eta_min + (eta_max - eta_min) * cosine_decay) / eta_init

    else:
        raise ValueError("Either 'eta_max' or 'growth_rate' must be provided.")


def get_lr_scheduler(optimizer, config, total_steps):
    lr_method = get_config_value(config, "lr_method")

    if lr_method == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        lr_step_type = "epoch"
    elif lr_method == "diminishing":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, diminishing_lr_lambda)
        lr_step_type = "step"
    elif lr_method == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_config_value(config, "epochs"), eta_min=config.get("lr_min", 0))
        lr_step_type = "epoch"
    elif lr_method == "poly":
        power = get_config_value(config, "power")
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=power)
        lr_step_type = "step"
    elif lr_method == "linear":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: linear_lr_lambda(step, total_steps))
        lr_step_type = "step"
    elif lr_method == "exp_growth":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: exp_growth_lr_lambda(
            epoch,
            eta_init=get_config_value(config, "init_lr"),
            incr_interval=get_config_value(config, "incr_interval"),
            epochs=get_config_value(config, "epochs"),
            eta_max=config.get("lr_max"),
            lr_growth_rate=config.get("lr_growth_rate")
        ))
        lr_step_type = "epoch"
    elif lr_method == "warmup_const":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: exp_warmup_const_lr_lambda(
            epoch,
            warmup_epochs=get_config_value(config, "warmup_epochs"),
            warmup_interval=get_config_value(config, "warmup_interval"),
            eta_init=get_config_value(config, "init_lr"),
            eta_max=config.get("lr_max"),
            lr_growth_rate=config.get("lr_growth_rate")
        ))
        lr_step_type = "epoch"
    elif lr_method == "warmup_cosine":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: exp_warmup_cosine_lr_lambda(
            epoch,
            warmup_epochs=get_config_value(config, "warmup_epochs"),
            warmup_interval=get_config_value(config, "warmup_interval"),
            epochs=get_config_value(config, "epochs"),
            eta_init=get_config_value(config, "init_lr"),
            eta_min=config.get("lr_min", 0),
            eta_max=config.get("lr_max"),
            lr_growth_rate=config.get("lr_growth_rate")
        ))
        lr_step_type = "epoch"
    else:
        raise ValueError(f"Unknown learning rate method: {lr_method}")

    return scheduler, lr_step_type
