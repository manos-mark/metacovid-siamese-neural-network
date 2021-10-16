from skimage import data, img_as_float
from skimage import exposure
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def histogram_equalization(img):
    """
    """
    
    eq_image = exposure.equalize_hist(img)
    
    return eq_image

