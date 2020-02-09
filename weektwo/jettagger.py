import h5py
import torch
import numpy as np
import gc
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import time
import json
from torchsummary import summary
import torchvision

from utils import read_data, iterate_minibatches
from utils import save_results

from custom_layers import Flatten, ConvBN2d, PConv2d, ConvBlock


class JetTagger:
    """Wrapper class for the construction, trianing and inference of tagging
    jets. This was written for the second Kaggle competition during the Yandex
    ML course at Imperial 2020.
    """
    N_DATA_SPLITS: int = 9
    batch_size: int = 1024
    number_of_chunks: int = 9
    VOLS_PREFIX: str = "/vols/cms/dmonk/mlatimperial/"

    def __init__(self, N_DATA_SPLITS=9, verbose=True) -> None:
        """ Class constructor.

        Parameters
        ----------
        N_DATA_SPLITS : int
            Number of data chunks to load for use in training the model. Will
            throw an error if greater than the number of available chunks.
        verbose: bool
            Set the verbosity of the output to stdout.
        """
        self.verbose: bool = verbose
        self.X_val = np.concatenate(
            [np.load(self.VOLS_PREFIX + "X_val_{}.npy".format(i)) for i in range(N_DATA_SPLITS)])
        self.y_val = np.concatenate(
            [np.load(self.VOLS_PREFIX + "y_val_{}.npy".format(i)) for i in range(N_DATA_SPLITS)])

    def _construct(self, config) -> torch.nn.Module:
        """ Internal function used to construct the model from a list of
        parameters.

        Parameters
        ----------
        config : list of dicts
            List of parameters in dict format.

        Returns
        -------
        torch.nn.Module
            Model of the neural net described in the configuration.
        """
        model: torch.nn.Module = torch.nn.Sequential()
        for index, layer in enumerate(config):
            model.add_module(
                "%s-%d" % (layer['type'].split(".")[-1], index),
                eval(layer['type'])(**layer['params']))
        return model

    def initModel(self, config_file) -> None:
        """ Initialise the model.

        Function also defines the optimiser and transformer for the model.
        # TODO: add hyperparameters for the optimiser and transformer to the
        JSON configuration file.

        Parameters
        ----------
        config_file : str
            Path to JSON file containig the parameters for the model.
        """
        self.device: torch.device = torch.device("cuda", 0)
        config_dict: dict = json.load(open(config_file))
        self.model: torch.nn.model = self._construct(config_dict["model"])
        self.model.to(self.device)
        if (self.verbose):
            print(summary(self.model, (3, 64, 64)))

        self.optimizer = eval(config_dict["optimizer"]["type"])(
            self.model.parameters(),
            **config_dict["optimizer"]["params"])

        self.transform = torchvision.transforms.RandomErasing(
            p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3),
            value=0, inplace=True)

    def saveModel(self, suffix="000") -> None:
        """ Save trained model to file.

        This function saves the entire model to file, rather than just the
        weights, resulting in a larger storage footprint.

        Parameters
        ----------
        suffix : str
            String to add to the end of the file to denote the current run,
            delimited by an underscore.
        """
        with open(os.path.join(self.VOLS_PREFIX, "nn_snapshots", "best_%s.pt" % suffix), 'wb') as f:
            torch.save(self.model, f)

    def train(self, num_epochs=2, batch_size=1024, number_of_chunks=9, suffix="000") -> None:
        """ Train the model.

        Parameters
        ----------
        num_epochs : int
            Number of epochs over which to train model.
        batch_size : int
            Number of concurrent models on which to train the model. This is
            generally limited by the size of the coprocessor memory.
        number_of_chunks : int
            Number of data chunks to use for training.
        suffix : str
            String to add to the end of the file to denote the current run,
            delimited by an underscore.
        """
        auc_history: list[float] = []
        best_score: float = 0
        best_epoch: int = 0

        # Loop for each epoch.
        for epoch in range(num_epochs):
            train_err = train_acc = 0
            train_batches = 0
            start_time = time.time()

            # Loop over given number of chunks.
            for step in range(number_of_chunks):
                X_train = np.load(self.VOLS_PREFIX + "X_train_{}.npy".format(step))
                y_train = np.load(self.VOLS_PREFIX + "y_train_{}.npy".format(step))
                train_batches += np.ceil(len(X_train) / batch_size).astype(int)
                self.model.train(True)  # <- set model in training mode
                # Loop over all batches in a chunk.
                for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                    X_batch = torch.FloatTensor(X_batch).to(self.device)
                    y_batch = torch.FloatTensor(y_batch).to(self.device)

                    # Transform data by randomly setting part of the image to
                    # zero.
                    X_batch = torch.stack(
                        [self.transform(img) for img in X_batch]
                    )

                    y_predicted = self.model(X_batch)
                    loss = torch.nn.functional.binary_cross_entropy(y_predicted, y_batch).mean()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_err += loss.data.cpu().numpy()
                    train_acc += torch.eq(torch.round(y_predicted), y_batch).data.cpu().numpy().mean()

                # Free up memory
                del X_train
                del y_train
                torch.cuda.empty_cache()

            # Make predictions using trained model.
            y_pred = []
            self.model.train(False)  # <- set model in prediciton mode
            for X_batch, y_batch in iterate_minibatches(self.X_val, self.y_val, batch_size, shuffle=False):
                X_batch = torch.FloatTensor(X_batch).to(self.device)
                y_pred.extend(self.model(X_batch).data.cpu().numpy())

            y_pred = np.asarray(y_pred)
            # Save the metrics values
            val_acc = accuracy_score(self.y_val, y_pred > 0.5)
            val_roc_auc = roc_auc_score(self.y_val, y_pred)
            auc_history.append(val_roc_auc)

            if (self.verbose):
                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))

                print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
                print("  train accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
                print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))
                print("  validation roc_auc:\t\t{:.2f} %".format(val_roc_auc * 100))

            # Save model if it is better than previous best
            if auc_history[-1] > best_score:
                best_score = auc_history[-1]
                best_epoch = epoch
                self.saveModel(suffix=suffix)

            # If the current and previous models obtain worse scores than the
            # one before, end training.
            elif (auc_history[-1] < auc_history[-2] and auc_history[-1] < auc_history[-3]):
                break

    def loadModel(self, filename, verbose=True) -> None:
        """ Load trained model.

        Parameters
        ----------
        filename : str
            Path to saved model.
        verbose : bool
            If set to true, a summary of the model is printed to stdout.
        """
        self.device = torch.device("cuda", 0)
        self.model = torch.load(filename)
        self.model.to(self.device)
        if (self.verbose):
            print(summary(self.model, (3, 64, 64)))

    def predict(self, suffix="000", chunk_size=1000, verbose=False) -> None:
        """ Use trained model to make predicitons for test data.

        Parameters
        ----------
        suffix : str
            String to add to the end of the file to denote the current run,
            delimited by an underscore.
        chunk_size : int
            Size of chunks to be processed in parallel.
        verbose : bool
            If set to true, get a more detailed output when loading chunks.
        """
        test = h5py.File(os.path.join(self.VOLS_PREFIX, "kaggle_test.h5"), 'r')
        y_ans = []
        self.model.train(False)
        for index, step in enumerate(range(0, len(test['all_events']['histHCAL']), chunk_size)):
            X, _ = read_data(test, False, step, step + chunk_size)
            y_ans.extend(self.model(torch.FloatTensor(X).to(self.device)).detach().cpu().numpy())
            del X
            gc.collect()
            if (verbose):
                print("Done:{}".format(index))

        y_ans = np.array(y_ans)
        save_results('{}'.format("predictions_%s.csv" % suffix), y_ans)
