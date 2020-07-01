import sys
sys.path.append("../")
from src.config import load_config
import argparse
from src.experiment import Experiment
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", type=str, help="Experiment configuration file")
args = parser.parse_args()
config = load_config(args.config)


def main():
    # Initializing wandb config
    #wandb.init(project="GWD", id=config["id"], group="Fold0", config=config, resume="allow")

    # Create Experiment
    experiment = Experiment(config)

    # Attach WandB to Experiment
    #experiment.attach_wandb(wandb)

    # Load checkpoint
    experiment.load_checkpoint()
    # Run Experiment
    # import torch
    # checkpoint  = experiment.runner.model.state_dict()
    # torch.save(checkpoint, "checkpoint/check14.pth")
    experiment.run()



if __name__ == "__main__":
    main()
    print("done")
