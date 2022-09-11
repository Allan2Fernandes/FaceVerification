import numpy
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

class ImageHandler:
    def __init__(self, relative_path, input_shape):
        self.relative_path = relative_path
        self.set_image_array()
        self.input_shape = input_shape
        pass

    def set_image_array(self):
        img = np.array(mpimg.imread(self.relative_path))
        self.image_array = img
        pass

    def get_image_array(self):
        image_array = np.resize(self.image_array, self.input_shape)
        #image_array = image_array/255.0
        #image_array = image_array.astype(numpy.half)
        return image_array

    def plotImage(self):
        plt.imshow(self.image_array)
        plt.show()
        pass