import sys

from jettagger import JetTagger

import warnings
warnings.filterwarnings(
    "ignore", message="Couldn't retrieve source code for container of type")


DATA_PREFIX = "data/"
VOLS_PREFIX = "/vols/cms/dmonk/mlatimperial/"


def main() -> None:
    # train = h5py.File(os.path.join(DATA_PREFIX, "kaggle_train.h5"), 'r')
    # save_data_in_chunks(train, 50000)

    suffix = "208"
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    batch_size = 64

    jt = JetTagger()
    jt.initModel("/home/hep/dm2614/projects/mlatimperial/submod_config.json")
    jt.train(
        num_epochs=num_epochs, number_of_chunks=9, batch_size=batch_size,
        suffix=suffix)
    # jt.loadModel("/vols/cms/dmonk/mlatimperial/nn_snapshots/best_208.pt")
    jt.predict(chunk_size=batch_size, suffix=suffix)


if __name__ == '__main__':
    main()
