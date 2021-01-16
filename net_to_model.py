#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess
from net import Net
""" 
argparser = argparse.ArgumentParser(description='Convert net to model.')
argparser.add_argument('net',
                       type=str,
                       help='Net file to be converted to a model checkpoint.')
argparser.add_argument('--start',
                       type=int,
                       default=0,
                       help='Offset to set global_step to.')
argparser.add_argument('--cfg',
                       type=argparse.FileType('r'),
                       help='yaml configuration with training parameters')
argparser.add_argument('-e',
                       '--ignore-errors',
                       action='store_true',
                       help='Ignore missing and wrong sized values.')
args = argparser.parse_args()
 """

model_file = "tf/128x10-t60-2-5300.pb.gz"
cfg_file = "tf/configs/example.yaml"

with open(cfg_file, "rb") as f:
    cfg = f.read()

cfg = yaml.safe_load(cfg)
print(yaml.dump(cfg, default_flow_style=False))
START_FROM = 0

tfp = tfprocess.TFProcess(cfg)
tfp.init_net_v2()
tfp.replace_weights_v2(model_file)
tfp.global_step.assign(START_FROM)

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
tfp.manager.save(checkpoint_number=START_FROM)
