import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np

import sys
# stdout = sys.stdout
stderr = sys.stderr
# sys.stdout = open('/dev/null', 'w')
sys.stderr = open('/dev/null', 'w')
# import keras
# sys.stdout = stdout

# import tensorflow as tf
from sample_models import *

# from data_generator import AudioGenerator
# from keras import backend as K
# from utils import int_sequence_to_text
# import sys
# from keras.backend.tensorflow_backend import set_session

def get_predictions():
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """

    print("OK1");
    print("OK2");
    return;



def main():

    # if (argv)
    # argv_1 = argv[1]
    # print("argv_1: {}".format(argv_1))
    # print("argv_1: {}".format( len(argv) )
    # print("argv, len:{}".format(len(argv)))
    # print("argv:{}".format(argv))

    # print(get_predictions)

    #
    get_predictions()

if __name__ == '__main__':
  main()
