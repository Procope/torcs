import os
import neat
import gzip
import random
import pickle
import numpy as np
import psutil
import importlib
import subprocess
from pytocl.protocol import Client
import my_driver
from my_driver import MyDriver
import neat
from neat.nn import FeedForwardNetwork


#################################################
with open('network_best.pickle', 'rb') as net_in:
    net = pickle.load(net_in)

# How to save a network
node_evals = [(node, bias, response, links) for (node, act_func, agg_func, bias, response, links) in net.node_evals]
with open('node_evals.pickle', 'wb') as f:
    pickle.dump(node_evals, f)
#################################################


def forward(filename, inputs):
    with open(filename, 'rb') as f:
        net_data = pickle.load(f)

    output_nodes = [0, 1]
    values = {}
    for k, v in enumerate(inputs, start=2):
        values[k] = v

    for (node, bias, response, links) in net_data:
        node_inputs = []

        for i, w in links:
            if i not in output_nodes:
                i = i - 10 if i > 0 else abs(i) + 1
                node_inputs.append(values[i] * w)

            # print(node, values[idx] * w)

        if node not in output_nodes:
            node = node - 10 if node > 0 else abs(node) + 1

        values[node] = np.tanh(bias + response * np.sum(node_inputs))

    return [values[i] for i in output_nodes]
