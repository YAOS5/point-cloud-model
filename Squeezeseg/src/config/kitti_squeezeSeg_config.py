# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def kitti_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')
  
  print("Kitti preloaded =", mc.LOAD_PRETRAINED_MODEL)
  print("Loaded model path = ", mc.PRETRAINED_MODEL_PATH)
  
  mc.CLASSES            = ['background', 'car']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  # controlling the relative importance of weights
  mc.CLS_LOSS_WEIGHT    = np.array([1.0/15.0, 1.0])
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
                                    [ 0.12,  0.56,  0.37]])


  mc.BATCH_SIZE         = 32
  # how many pixel long our projection is
  mc.AZIMUTH_LEVEL      = 512
  # how many pixel tall our projection is
  mc.ZENITH_LEVEL       = 64

  # Conditional Random Field
  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  # How many times it is passed through the RNN
  mc.RCRF_ITER          = 3
  mc.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6])
  mc.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = np.array([.9, .9, .6, .6])
  mc.ANG_FILTER_COEF    = 0.02

  # Loss coefficient
  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.00001
  mc.LEARNING_RATE      = 0.0001
  mc.DECAY_STEPS        = 100
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.1

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  #mc.INPUT_MEAN         = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
  #mc.INPUT_STD          = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

  # x, y, z, intensity, distance
  mc.INPUT_MEAN         = np.array([[[0, 0, 5, 0.5, 6]]])
  mc.INPUT_STD          = np.array([[[1, 1, 1, 1, 1]]])

  return mc
