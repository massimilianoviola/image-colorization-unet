import os
import yaml

# path to the YAML config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH, "r") as file:
    CONFIG = yaml.safe_load(file)
