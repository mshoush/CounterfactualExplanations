import sys
import os

# Add the counterfactual-search directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the build_model function from src.models
from src.models import build_model

# # Use the build_model function as needed
# if __name__ == "__main__":
#     model = build_model('classification')
#     print("Model built successfully:", model)

print("Here....")
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.models import build_model

import argparse
import logging
import os
import shutil

import sys
import os

import os
import sys

from src.models import build_model

import yaml
from easydict import EasyDict as edict

from src.models import build_model
from src.trainers import build_trainer
from src.utils.generic_utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=False, help='Configuration file path to start training from scratch')
parser.add_argument('-cp', '--continue_path', type=str, required=False, help='Path to the existing training run to continue interrupted training')
opt = parser.parse_args()

print(f"\nopt: {opt}\n")

def main(args):
    print(f"\nargs: {args}\n")
    with open(args.config_path or os.path.join(args.continue_path, 'hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    model = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size)
    trainer = build_trainer(opt.task_name, opt, model, args.continue_path)

    if args.continue_path is None:
        shutil.copy2(args.config_path, trainer.logging_dir / 'hparams.yaml')
    logging.info('Started training.')
    trainer.fit()
    logging.info('Finished training.')


if __name__ == '__main__':
    main(parser.parse_args())
