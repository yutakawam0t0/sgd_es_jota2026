def get_config_value(config, key):
    value = config.get(key)
    if value is None:
        raise ValueError(f"The '{key}' parameter is missing in the config.")
    return value
