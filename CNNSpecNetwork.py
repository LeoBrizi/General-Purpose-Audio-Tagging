import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras.models as KM
import tensorflow.keras.utils as KU
import tensorflow.keras.callbacks as KC


class CNNSpecNetwork():

    def __init__(self, num_of_classes, spec_bins, n_frames, num_filters=48):
        init_conv = keras.initializers.he_normal(seed=None)
        activation_conv = keras.activations.relu(
            x, alpha=0.0, max_value=None, threshold=0.0)
        self.n_frames = n_frames  # 384
        self.spec_bins = spec_bins  # 128
        self.num_filters = num_filters
        self.num_of_classes = num_of_classes
        self.use_validation_set = False
        input_shape = (1, self.spec_bins, self.n_frames)
        self.model = K.Sequential()

        # --- conv layers ---
        self.model.add(KL.Conv2D(self.num_filters, (5, 5), strides=(2, 2), padding='same',
                                 kernel_initializer=init_conv, activation=activation_conv, input_shape=input_shape))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Conv2D(self.num_filters, (3, 3), strides=(
            1, 1), padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.MaxPooling2D(pool_size=2))
        self.model.add(KL.Dropout(0.3))

        self.model.add(KL.Conv2D(2 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Conv2D(2 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.MaxPooling2D(pool_size=2))
        self.model.add(KL.Dropout(0.3))

        self.model.add(KL.Conv2D(4 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Dropout(0.3))
        self.model.add(KL.Conv2D(4 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Dropout(0.3))
        self.model.add(KL.Conv2D(6 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Dropout(0.3))
        self.model.add(KL.Conv2D(6 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.MaxPooling2D(pool_size=2))
        self.model.add(KL.Dropout(0.3))

        self.model.add(KL.Conv2D(8 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Conv2D(8 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.MaxPooling2D(pool_size=(1, 2)))
        self.model.add(KL.Dropout(0.3))

        self.model.add(KL.Conv2D(8 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Conv2D(8 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='same', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.MaxPooling2D(pool_size=(1, 2)))
        self.model.add(KL.Dropout(0.3))

        self.model.add(KL.Conv2D(8 * self.num_filters, (3, 3), strides=(1, 1),
                                 padding='valid', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Dropout(0.5))
        self.model.add(KL.Conv2D(8 * self.num_filters, (1, 1), strides=(1, 1),
                                 padding='valid', kernel_initializer=init_conv, activation=activation_conv))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Dropout(0.5))

        # --- feed forward part ---
        self.model.add(KL.Conv2D(self.num_of_classes, kernel_size=1,
                                 kernel_initializer=init_conv, activation=None))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.GlobalAveragePooling2D())
        self.model.add(KL.Flatten())
        self.model.add(Activation("softmax"))

    def compile_model(self, learning_rate, decay=0.0):
        self.optimizer = KO.Adam(lr=learning_rate, decay=decay)
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy',
                           metrics={'accu': acc, 'loss': loss})
        return self.model

    def load_model(self, model_name):
        del self.model
        self.model = KM.load_model("Models/" + model_name + ".h5")
        print("Loaded model from disk")

    def save_model(self, model_name):
        self.model.save("Models/" + model_name + ".h5")
        print("Saved model to disk")

    def get_summary(self, model_name):
        print(self.model.summary())
        KU.plot_model(self.model, to_file="Models/" + model_name +
                      ".png", show_shapes=True, show_layer_names=True)
        print("Saved a picture of the model")

    def fit(self, data, labels, model_name, learning_rate, batch_size=32, epochs=10, patience=5, callbacks=[], validation_split=0.2, verbose=2):
        from datetime import datetime
        callbacks = []

        if(validation_split != 0):
            self.use_validation_set = True

        def scheduler(epoch):
            return learning_rate

        learning_rate_callback = KC.LearningRateScheduler(scheduler)
        callbacks.append(learning_rate_callback)

        logdir = "Models/" + model_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = KC.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

        if(callbacks == [] or 'check_pointer' in callbacks):
            check_pointer_callback = KC.ModelCheckpoint(
                filepath="Models/" + model_name + "/tmp/weights.h5", monitor='val_acc', verbose=1, save_best_only=True)
            callbacks.append(check_pointer_callback)

        if(callbacks == [] or 'early_stop' in callbacks):
            early_stop_callback = KC.EarlyStopping(
                monitor='val_acc', patience=patience, mode='auto')
            callbacks.append(early_stop_callback)

        self.history = self.model.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                                      callbacks=callbacks, validation_split=validation_split, verbose=verbose)
        return self.history

    def predict(self, data):

        label_pred = self.model.predict(data)
        return label_pred

    def print_history(self, model_name):
        import matplotlib.pyplot as plt

        plt.plot(self.history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        if self.use_validation_set:
            plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("Models/" + model_name + "_acc.png")
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        if self.use_validation_set:
            plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("Models/" + model_name + "_loss.png")
        plt.close()
