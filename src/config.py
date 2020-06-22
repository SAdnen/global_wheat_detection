from pathlib import Path
import yaml


root = Path.cwd().parent
train_dir = root/"train"
test_dir = root/"test"
checkpoint_dir = root/"experiment/checkpoint"

path_settings = {"train": train_dir,
                 "test": test_dir,
                 "checkpoint": checkpoint_dir}


def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
