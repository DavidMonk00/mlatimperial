import h5py
import torch
import numpy as np
import gc
import pickle
import os
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from tqdm import trange
import time
import pandas as pd
import sys


def read_data(data, is_train=True, start_ind=0, end_ind=0):
    layer_hcal = np.expand_dims(
        data['all_events']['histHCAL'][start_ind:end_ind], -1
    ).astype(np.float32)
    layer_em = np.expand_dims(
        data['all_events']['histEM'][start_ind:end_ind], -1
    ).astype(np.float32)
    layer_track = np.expand_dims(
        data['all_events']['histtrack'][start_ind:end_ind], -1
    ).astype(np.float32)

    hit_map = np.concatenate(
        (layer_hcal, layer_em, layer_track), axis=-1).astype(np.float32)
    hit_map = np.rollaxis(hit_map, 3, 1)
    hit_map = (hit_map - hit_map.mean(axis=0, keepdims=True)) / \
        hit_map.std(axis=0, keepdims=True)
    answers = None
    if is_train:
        answers = np.expand_dims(
            data['all_events']['y'][start_ind:end_ind], -1)
    return hit_map, answers


def save_data_in_chunks(data, chunk_size):
    for index, step in enumerate(range(0, len(data['all_events']['histHCAL']), chunk_size)):
        X, y = read_data(data, True, step, step + chunk_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=0.9, random_state=42)
        np.save("X_train_{}".format(index), X_train)
        np.save("y_train_{}".format(index), y_train)
        np.save("X_val_{}".format(index), X_val)
        np.save("y_val_{}".format(index), y_val)
        del X, y, X_train, X_val, y_train, y_val
        gc.collect()
        print("Done:{}".format(index))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def iterate_minibatches(X, y, batchsize, shuffle=False):
    indices = np.arange(len(X))
    if shuffle:
        indices = np.random.permutation(indices)
    for start in trange(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield X[ix], y[ix]


def save_results(filepath, y_ans):
    answer_dataframe = pd.DataFrame(columns=["ID", "ans"])
    answer_dataframe['ID'] = range(0, len(y_ans))
    answer_dataframe['ans'] = y_ans
    answer_dataframe.to_csv(filepath, index=False)
