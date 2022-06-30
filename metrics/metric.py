import imp
from summary import ComputeLayerwiseSummary
from size import ComputeSize
from params import ComputeParams
from flops import ComputeFlops
from timer import ComputeExecutionTime

import argparse
from json import load
from typing import Tuple
from xmlrpc.client import Boolean
import tensorflow as tf


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument(
        '--model_dir',
        default=None,
        type=str,
        required=True,
        help='Saved Keras model location'
        )
    #  parser.add_argument(
    #     '--arch',
    #     default=None,
    #     type=str,
    #     required=True,
    #     help='Backbone architecture type'
    #  )
        parser.add_argument(
        '--flops',
        default=True,
        type=Boolean,
        required=False,
        help='Is Flops calculation needed'
        )

        parser.add_argument(
        '--layer_summary',
        default=True,
        type=Boolean,
        required=False,
        help='Is Layer summary needed '
        )

        parser.add_argument(
        '--mem_size',
        default=True,
        type=Boolean,
        required=False,
        help='Is memory consumption and size needed'
        )

        parser.add_argument(
        '--num_params',
        default=True,
        type=Boolean,
        required=False,
        help='Is param count calculation needed'
        )

        parser.add_argument(
        '--time',
        default=True,
        type=Boolean,
        required=False,
        help='Is inference time calculation needed'
        )

        parser.add_argument(
        '--img_ht',
        default=True,
        type=int,
        required=True,
        help='Height of image size(H,W)'
        )
        parser.add_argument(
        '--img_wd',
        default=True,
        type=int,
        required=True,
        help='Width of image size(H,W)'
        )

        args = parser.parse_args()

        #Load model
        model = load()
        #Image shape
        height = args.img_ht
        width = args.img_wd
        img_shape = (height, width)

        result = {}

        if args.layer_summary == True:
            summary = ComputeLayerwiseSummary()
            result['summary'] = summary(model)

        if args.mem_size == True:
            size = ComputeSize()
            mem_size = size(model)
            result['memory_footprint'] = mem_size['memory_footprint']
            result['model_size'] = mem_size['model_size']
        
        if args.num_params == True:
            params = ComputeParams()
            result['num_params'] = params(model)
        
        if args.time == True:
            timer = ComputeExecutionTime()
            exectime = timer(model, input_shape=img_shape)
            result['execution_time'] = exectime
        
        if args.flops == True:
            flops = ComputeFlops()
            result['flops'] = flops(model, input_shape=img_shape)
            
