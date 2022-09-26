from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
import numpy

class Prebuilt_Network:
    def __init__(self, model, train_left_array, train_right_array, train_label_array, val_left_array, val_right_array, val_label_array, save_path):
        self.model = model
        self.train_left_array = train_left_array
        self.train_right_array = train_right_array
        self.train_label_array = numpy.array(train_label_array, dtype=float)
        self.val_left_array = val_left_array
        self.val_right_array = val_right_array
        self.val_label_array = numpy.array(val_label_array, dtype=float)
        self.save_path = save_path
        self.lowest_loss = 10
        train_datagen = ImageDataGenerator(rescale= 1./255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
        validation_datagen = ImageDataGenerator(rescale=1./255)
        self.train_generator = train_datagen.flow(x = [self.train_left_array, self.train_right_array], y = self.train_label_array, batch_size = 32, shuffle=True)
        self.validation_generator = validation_datagen.flow(x = [self.val_left_array, self.val_right_array], y = self.val_label_array, batch_size = 8)
        print(f"Images of training generator have shape: {self.train_generator.x.shape}")
        print(f"Labels of training generator have shape: {self.train_generator.y.shape}")
        pass

    def early_stop_callback(self, batch, logs):
        current_loss = logs.get('val_loss')

        if (current_loss < self.lowest_loss and current_loss < 0.45):
            self.lowest_loss = current_loss
            self.save_the_model()
            print("Saved the model")
            pass

        if current_loss >= 1.2 * self.lowest_loss:
            self.model.stop_training = True

            pass
        pass


    def show_model_summary(self):
        self.model.summary()
        pass

    def train_the_model(self):
        my_callback = callbacks.LambdaCallback(on_epoch_end=self.early_stop_callback)
        self.model.fit(self.train_generator, epochs=100, validation_data = self.validation_generator, callbacks=[my_callback])
        pass

    def validate_the_model(self, batch, logs):
        self.model.evaluate([self.val_left_array, self.val_right_array], self.val_label_array, batch_size=32)
        pass

    def save_the_model(self):
        self.model.save(self.save_path)
        pass

    def get_the_model(self):
        return self.model
