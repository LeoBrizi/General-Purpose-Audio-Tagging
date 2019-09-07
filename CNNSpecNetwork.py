import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras.models as KM
import tensorflow.keras.utils as KU
import tensorflow.keras.callbacks as KC
import tensorflow.compat.v1.keras.initializers as KI
import tensorflow.keras.activations as KA
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CNNSpecNetwork():

    def __init__(self, n_frames, spec_bins=128, num_of_classes=41, num_filters=48):
        init_conv = KI.he_normal(seed=None)
        activation_conv = KA.relu
        self.n_frames = n_frames
        self.spec_bins = spec_bins
        self.num_filters = num_filters
        self.num_of_classes = num_of_classes
        self.use_validation_set = False
        input_shape = (self.spec_bins, self.n_frames, 1)
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
        self.model.add(KL.Activation("softmax"))

    def compile_model(self, learning_rate, decay=0.0):
        self.optimizer = KO.Adam(lr=learning_rate, decay=decay)
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model

    def load_model(self, model_name):
        del self.model
        self.model = KM.load_model(
            "Models/" + model_name + "/" + model_name + ".h5")
        print("Loaded model from disk")

    def save_model(self, model_name):
        import os
        if not os.path.exists("Models/" + model_name):
            os.makedirs("Models/" + model_name)
        self.model.save("Models/" + model_name + "/" + model_name + ".h5")
        print("Saved model to disk")

    def get_summary(self, model_name):
        print(self.model.summary())
        import os
        if not os.path.exists("Models/" + model_name):
            os.makedirs("Models/" + model_name)
        KU.plot_model(self.model, to_file="Models/" + model_name + "/" +
                      model_name + ".png", show_shapes=True, show_layer_names=True)
        print("Saved a picture of the model")

    def fit(self, data, labels, model_name, learning_rate, early_stop, batch_size=32, epochs=10, patience=5, validation_split=0.2, verbose=2):
        from datetime import datetime
        import os
        callbacks = []

        if(validation_split != 0):
            self.use_validation_set = True

        def scheduler(epoch):
            return learning_rate

        learning_rate_callback = KC.LearningRateScheduler(scheduler)
        callbacks.append(learning_rate_callback)

        logdir = "Models/" + model_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        check_pointer_callback = KC.ModelCheckpoint(
            filepath=logdir + "/weights.h5", monitor='val_acc', verbose=1, save_best_only=True)
        callbacks.append(check_pointer_callback)

        if(early_stop):
            early_stop_callback = KC.EarlyStopping(
                monitor='val_acc', patience=patience, mode='auto')
            callbacks.append(early_stop_callback)

        history = self.model.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                                 callbacks=callbacks, validation_split=validation_split, verbose=verbose)
        return history

    def predict_single_data(self, data):
        start = 0
        outputs = []
        while(start + self.n_frames <= data.shape[1]):
            inp = np.expand_dims(data[:, start:start + self.n_frames], axis=0)
            inp = np.expand_dims(inp, axis=3)
            pred = self.model.predict(inp)
            outputs.append(pred)
            start += self.n_frames
        res = np.zeros(self.num_of_classes)
        for output in outputs:
            res += np.squeeze(output)
        res /= len(outputs)
        return np.argmax(res), np.max(res)

    def print_history(self, model_name, history):
        import matplotlib.pyplot as plt
        self.history = history
        plt.plot(history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        if self.use_validation_set:
            plt.plot(history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("Models/" + model_name + "_acc.png")
        plt.close()
        # summarize history for loss
        plt.plot(history['loss'])
        if self.use_validation_set:
            plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("Models/" + model_name + "_loss.png")
        plt.close()

    def save_history(self, model_name, history):
        import json
        jsond = json.dumps(str(history))
        f = open("Models/" + model_name + "/history.json", "w")
        f.write(jsond)
        f.close()

    def load_history(self, model_name):
        import json
        with open("Models/" + model_name + "/history.json", 'r') as f:
            history = json.load(f)
            f.close()
        return history

    def compute_stat(self, X_data):
        output_vec = []
        class_dict = {}
        how_many_per_class = {}
        for X in tqdm(X_data):
            clas, output = self.predict_single_data(X)
            output_vec.append((clas, output))
            class_dict.setdefault(clas, {'mean': 0, 'std_dev': 0})
            how_many_per_class[
                clas] = how_many_per_class.setdefault(clas, 0) + 1
            Xk = output
            k = how_many_per_class[clas]
            delta = (Xk - class_dict[clas]["mean"]) / k
            class_dict[clas]["mean"] = class_dict[
                clas]["mean"] + delta
            class_dict[clas]["std_dev"] = ((
                (k - 1) * class_dict[clas]["std_dev"]) / k) + (delta * (Xk - class_dict[clas]["mean"]))
        for key in class_dict.keys():
            class_dict[key]['std_dev'] = np.sqrt(class_dict[key]['std_dev'])

        return output_vec, class_dict, how_many_per_class
