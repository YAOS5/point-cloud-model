# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def kitti_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')
  
  print("Kitti preloaded =", mc.LOAD_PRETRAINED_MODEL)
  
  mc.CLASSES            = ['unknown', 'car']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([0.5, 1.0])
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00], [ 0.12,  0.56,  0.37]])

  mc.BATCH_SIZE         = 4
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  mc.BILATERAL_THETA_A  = np.array([.9, .9])
  mc.BILATERAL_THETA_R  = np.array([.015, .015])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = np.array([.9, .9])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 10.0
  mc.WEIGHT_DECAY       = 0.0
  mc.LEARNING_RATE      = 0.001
  mc.DECAY_STEPS        = 200
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance
  mc.INPUT_MEAN         = np.array([[[0, 0, 5, 1, 6]]])
  mc.INPUT_STD          = np.array([[[1, 1, 1, 1, 1]]])

  return mc
