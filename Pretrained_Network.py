import tensorflow
from tensorflow import keras
from keras import callbacks

class Pretrained_Network:
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
        my_callback = callbacks.LambdaCallback(on_epoch_end=self.validate_the_model)
        self.model.fit([self.train_left_array, self.train_right_array], self.train_label_array, epochs=1, shuffle=True,batch_size=8, callbacks=my_callback)
        pass

    def validate_the_model(self, batch, logs):
        self.model.evaluate([self.val_left_array, self.val_right_array], self.val_label_array, batch_size=32)
        pass

    def save_the_model(self):
        self.model.save(self.save_path)
        pass

    def get_the_model(self):
        return self.model
