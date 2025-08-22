import yaml

# ------------------- Config Load ------------------------

class Config:
    """
        Turns the dictionnary into an object with dot notation access
    """
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)