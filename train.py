import argparse

import yaml
from task import TrainingSimpleCNN
from model.simple_cnn import Simple_CNN
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)

args = parser.parse_args()

config = get_config(args.config_file)

model = Simple_CNN(config.MODEL)

task = TrainingSimpleCNN(config, model)
task.start()
task.get_predictions()
