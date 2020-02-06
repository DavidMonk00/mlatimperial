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
import json
import torch
from torchsummary import summary

from utils import read_data, save_data_in_chunks, Flatten, iterate_minibatches
from utils import save_results

import warnings
warnings.filterwarnings("ignore", message="Couldn't retrieve source code for container of type")


DATA_PREFIX = "data/"
VOLS_PREFIX = "/vols/cms/dmonk/mlatimperial/"


def construct(config):
    model = torch.nn.Sequential()
    for index, layer in enumerate(config):
        model.add_module(
            "%s-%d" % (layer['type'].split(".")[-1], index),
            eval(layer['type'])(**layer['params']))
    return model


def main():
    # train = h5py.File(os.path.join(DATA_PREFIX, "kaggle_train.h5"), 'r')
    # save_data_in_chunks(train, 50000)

    N_DATA_SPLITS = 9
    X_val = np.concatenate(
        [np.load(VOLS_PREFIX + "X_val_{}.npy".format(i)) for i in range(N_DATA_SPLITS)])
    y_val = np.concatenate(
        [np.load(VOLS_PREFIX + "y_val_{}.npy".format(i)) for i in range(N_DATA_SPLITS)])

    device = torch.device("cuda", 0)

    model = construct(json.load(
        open("/home/hep/dm2614/projects/mlatimperial/config.json")))

    model.to(device)

    print(summary(model, (3, 64, 64)))

    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = int(sys.argv[1])  # amount of passes through the data
    batch_size = 1024  # number of samples processed at each function call
    auc_history = []

    number_of_chunks = 9  # number of initial data splits to process
    best_score = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:

        train_err = train_acc = 0
        train_batches = 0
        start_time = time.time()

        for step in range(number_of_chunks):
            X_train = np.load(VOLS_PREFIX + "X_train_{}.npy".format(step))
            y_train = np.load(VOLS_PREFIX + "y_train_{}.npy".format(step))
            train_batches += np.ceil(len(X_train) / batch_size).astype(int)
            # This is you have see already - traning loop
            model.train(True)  # enable dropout / batch_norm training behavior
            for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                X_batch = torch.FloatTensor(X_batch).to(device)
                y_batch = torch.FloatTensor(y_batch).to(device)

                y_predicted = model(X_batch)
                loss = torch.nn.functional.binary_cross_entropy(y_predicted, y_batch).mean()

                loss.backward()
                opt.step()
                opt.zero_grad()

                train_err += loss.data.cpu().numpy()
                train_acc += torch.eq(torch.round(y_predicted), y_batch).data.cpu().numpy().mean()

            del X_train
            del y_train
            torch.cuda.empty_cache()

        # And a full pass over the validation data:
        y_pred = []

        model.train(False)
        for X_batch, y_batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            X_batch = torch.FloatTensor(X_batch).to(device)
            y_pred.extend(model(X_batch).data.cpu().numpy())

        y_pred = np.asarray(y_pred)
        # Save the metrics values
        val_acc = accuracy_score(y_val, y_pred > 0.5)
        val_roc_auc = roc_auc_score(y_val, y_pred)
        auc_history.append(val_roc_auc)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
        print("  train accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))
        print("  validation roc_auc:\t\t{:.2f} %".format(val_roc_auc * 100))

        if auc_history[-1] > best_score:
            best_score = auc_history[-1]
            best_epoch = epoch
            with open(os.path.join(VOLS_PREFIX, "nn_snapshots", "best.pt"), 'wb') as f:
                torch.save(model, f)

    # chunk_size = 1000
    # test = h5py.File(os.path.join(VOLS_PREFIX, "kaggle_test.h5"), 'r')
    #
    # y_ans = []
    # for index, step in enumerate(range(0, len(test['all_events']['histHCAL']), chunk_size)):
    #     X, _ = read_data(test, False, step, step + chunk_size)
    #     y_ans.extend(model(torch.FloatTensor(X).to(device)).detach().cpu().numpy())
    #     del X
    #     gc.collect()
    #     print("Done:{}".format(index))
    #
    # y_ans = np.array(y_ans)
    # save_results(os.path.join(VOLS_PREFIX, '{}'.format("baseline.csv")), y_ans)


if __name__ == '__main__':
    main()
