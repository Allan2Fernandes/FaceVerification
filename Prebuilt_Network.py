from datetime import datetime
import tensorflow
from tensorflow import keras
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt

class Prebuilt_Network:
    def __init__(self, model, train_left_array, train_right_array, train_label_array, val_left_array, val_right_array, val_label_array, save_path):
        self.model = model
        self.train_left_array = train_left_array
        self.train_right_array = train_right_array
        self.train_label_array = train_label_array
        self.val_left_array = val_left_array
        self.val_right_array = val_right_array
        self.val_label_array = val_label_array
        self.save_path = save_path
        pass

    def show_model_summary(self):
        self.model.summary()
        pass

    def train_the_model(self):

        self.model.fit([self.train_left_array, self.train_right_array], self.train_label_array, epochs=50, shuffle=True,batch_size=16, validation_data = ([self.val_left_array, self.val_right_array], self.val_label_array))
        pass

    def validate_the_model(self, batch, logs):
        self.model.evaluate([self.val_left_array, self.val_right_array], self.val_label_array, batch_size=32)
        pass

    def save_the_model(self):
        self.model.save(self.save_path)
        pass

    def get_the_model(self):
        return self.model
