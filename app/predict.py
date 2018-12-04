import tensorflow as tf
import sys, os
import numpy as np
import cv2
CURDIR = os.path.dirname(os.path.realpath(__file__))
ROOTDIR = os.path.abspath(os.path.join(CURDIR, '..'))
sys.path.insert(0,ROOTDIR)
from dataset import prepare_single_test

# def as_ndarray(img_as_str):
#     '''Accepts request.data (string).
#     Returns jpeg as numpy.ndarray of uint8).

#     TODO: making assumption for now about color, channel order --
#         is this stored as metadata with the image?
#     '''
#     img = np.fromstring(img_as_str, np.uint8)
#     return cv2.imdecode(img, cv2.IMREAD_COLOR)

# def predict(model, img_as_str):
#     img = as_ndarray(img_as_str)
#     return model.test_single(model.sess, img, model.vocabulary, model.config)

def predict(M, img_path):
    data =  prepare_single_test(M.config, img_path)
    return M.model.test_single(M.sess, data, M.vocabulary)