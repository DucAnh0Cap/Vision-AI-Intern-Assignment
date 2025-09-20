import argparse

import yaml
from task.training_simple_cnn_task import TrainingSimpleCNN
from model.simple_cnn import Simple_CNN


parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)

args = parser.parse_args()

with open(args.config_file, 'rb') as f:
    config = yaml.safe_load(f)

model = Simple_CNN(config.MODEL)

task = TrainingSimpleCNN(config, model)
task.start()
task.get_predictions()
