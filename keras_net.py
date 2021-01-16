#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess
from net import Net
import tfprocess
from net import Net

class KerasNet:

    def __init__(self, model_file="128x10-t60-2-5300.pb.gz", cfg_file="configs/example.yaml"):

        with open(cfg_file, "rb") as f:
            cfg = f.read()

        cfg = yaml.safe_load(cfg)
        print(yaml.dump(cfg, default_flow_style=False))

        tfp = tfprocess.TFProcess(cfg, gpu=True)
        tfp.init_net_v2()
        tfp.replace_weights_v2(model_file)

        self.model = tfp.model 

    def evaluate(self, board):

        input_planes = board.lcz_features()
        model_input = input_planes.reshape(1, 112, 64)
        model_output = self.model.predict(model_input)

        policy = model_output[0][0]
        value = model_output[1]

        #TODO run model output through a softmax

        return model_output
