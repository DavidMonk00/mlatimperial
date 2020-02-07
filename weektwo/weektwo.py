import sys

from jettagger import JetTagger

from custom_layers import Flatten, ConvBN2d, PConv2d, ConvBlock

import warnings
warnings.filterwarnings(
    "ignore", message="Couldn't retrieve source code for container of type")


DATA_PREFIX = "data/"
VOLS_PREFIX = "/vols/cms/dmonk/mlatimperial/"


def main():
    # train = h5py.File(os.path.join(DATA_PREFIX, "kaggle_train.h5"), 'r')
    # save_data_in_chunks(train, 50000)

    suffix = "203"
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    jt = JetTagger()
    jt.initModel("/home/hep/dm2614/projects/mlatimperial/submod_config.json")
    jt.train(
        num_epochs=num_epochs, number_of_chunks=2, batch_size=128,
        suffix=suffix)
    jt.predict(suffix=suffix)


if __name__ == '__main__':
    main()
