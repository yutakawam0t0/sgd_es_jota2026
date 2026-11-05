import math
from optim.bs_scheduler import LambdaBS
from .get_config_value import get_config_value


def exp_growth_bs_lambda(epoch, b_0, incr_interval, bs_growth_rate, epochs, b_max=None):
    """Exponentially increases the batch size at each interval."""
    if b_max is not None:
        exponent = (epochs - incr_interval) / incr_interval
        a = (b_max - b_0) / (bs_growth_rate ** exponent - 1)
        b = b_0 - a
        return min((1 / b_0) * (a * (bs_growth_rate ** (epoch // incr_interval)) + b), b_max / b_0)
    else:
        return bs_growth_rate ** (epoch // incr_interval)


def get_bs_lambda(config):
    """Returns the batch size lambda function based on the configuration."""
    bs_method = get_config_value(config, "bs_method")

    if bs_method == "constant":
        return lambda epoch: 1

    elif bs_method == "exp_growth":
        return lambda epoch: exp_growth_bs_lambda(
            epoch,
            b_0=get_config_value(config, "init_bs"),
            incr_interval=get_config_value(config, "incr_interval"),
            bs_growth_rate=get_config_value(config, "bs_growth_rate"),
            epochs=get_config_value(config, "epochs"),
            b_max=config.get("bs_max")
        )
    else:
        raise ValueError(f"Unknown batch size method: {bs_method}")


def steps_per_epoch(config, trainset_length):
    """Calculates the number of steps per epoch based on the batch size method."""
    bs_lambda = get_bs_lambda(config)
    return lambda epoch: math.ceil(trainset_length / math.ceil(get_config_value(config, "init_bs") * bs_lambda(epoch)))


def calculate_total_steps(config, trainset_length):
    """Calculates the total number of steps for all epochs."""
    total_steps = sum(steps_per_epoch(config, trainset_length)(epoch) for epoch in range(get_config_value(config, "epochs")))
    return total_steps


def get_bs_scheduler(config, trainset_length):
    """Returns a batch size scheduler and the total number of steps."""
    bs_lambda = get_bs_lambda(config)
    bs_scheduler = LambdaBS(initial_bs=get_config_value(config, "init_bs"), bs_lambda=bs_lambda)
    total_steps = calculate_total_steps(config, trainset_length)

    return bs_scheduler, total_steps
