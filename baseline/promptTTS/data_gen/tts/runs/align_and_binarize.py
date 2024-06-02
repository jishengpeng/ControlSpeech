import utils.commons.single_thread_env  # NOQA
from data_gen.tts.runs.binarize import binarize
from data_gen.tts.runs.preprocess import preprocess

if __name__ == '__main__':
    preprocess()
    binarize()
