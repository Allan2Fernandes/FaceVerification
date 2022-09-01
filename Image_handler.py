import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

class ImageHandler:
    def __init__(self, relative_path):
        self.relative_path = relative_path
        self.set_image_array()
        pass

    def set_image_array(self):
        img = np.array(mpimg.imread(self.relative_path))
        self.image_array = img
        pass

    def get_image_array(self):
        return self.image_array

    def plotImage(self):
        plt.imshow(self.image_array)
        plt.show()
        pass