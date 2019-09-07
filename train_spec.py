import argparse
from CNNSpecNetwork import *
from DataLoader import *
import tensorflow.keras.utils as KU
from DataManager import *
from tqdm import tqdm

# ritorna un vettore di indici che sono quelli da scartare
EPOCHS_PER_TRAIN_F = 3
EPOCHS_PER_TRAIN_S = 5


def self_verify(labels, verified, predict, class_stat, max_per_class, loaded_per_class, threshold):
    indeces = []
    for index in tqdm(range(len(labels))):
        label = labels[index]
        if(verified[index]):
            if(loaded_per_class[label] < max_per_class[label]):
                loaded_per_class[label] += 1
                continue
        if(label == predict[index][0]):
            if(abs(predict[index][1] - class_stat[label]['mean']) < threshold * class_stat[label]['std_dev']):
                if(loaded_per_class[label] < max_per_class[label]):
                    loaded_per_class[label] += 1
                    continue
        indeces.append(index)
    return indeces, loaded_per_class


def main(model_name, continue_train, sample_rate, spec_version, dimension_conform, num_of_frame, threshold, pass_dataset,
         num_for_each_train, num_of_filter, first_phase_lp, second_phase_lp, validation_split, early_stopping, verbose):
    model = CNNSpecNetwork(n_frames=num_of_frame, num_filters=num_of_filter)
    model.compile_model(learning_rate=first_phase_lp['lr'])
    if(continue_train):
        model.load_model(model_name=model_name)
    if(verbose):
        print("MODEL SUMMARY:")
        model.get_summary(model_name=model_name)

    data_loader = DataLoader(sample_rate=sample_rate,
                             dim_to_conform=dimension_conform)
    data_loader.load_files_labels(file_csv="./dataset/train.csv")
    class_id_mapping = data_loader.get_label_id_mapping()
    classes_frequency, classes_percent, verif_num, verif_perc, classes_verified, classes_percent_verified = data_loader.get_general_statistics()
    random_slice = DataManager.get_prepare_random_slice(num_of_frame)

    if(verbose):
        print("DATASET STATISTICS:")
        print("how many files per classes")
        print(classes_frequency)
        print("--------------------------------------------------------------")
        print("how much percentage for class in the dataset")
        print(classes_percent)
        print("--------------------------------------------------------------")
        print("how many files are manually verified")
        print(verif_num)
        print("--------------------------------------------------------------")
        print("how much percentage of verified files ")
        print(verif_perc)
        print("--------------------------------------------------------------")
        print("how many files are manually verified per class")
        print(classes_verified)
        print("--------------------------------------------------------------")
        print("how much percentage of verified files per class")
        print(classes_percent_verified)
        print()
        print()

    self_verify_max_per_class = {}
    loaded_per_class = {}
    for key in classes_frequency.keys():
        non_verified = classes_frequency[key] - classes_verified[key]
        self_verify_max_per_class[class_id_mapping[key]] = classes_frequency[
            key] - (30 * non_verified / 100)
        loaded_per_class[class_id_mapping[key]] = 0
    # first train the model on the verified files
    X, Y = data_loader.load_verified_spectrograms(spec_version)
    files_used = len(Y)
    #Y = KU.to_categorical(Y)

    if(verbose):
        v = 2
    else:
        v = 1
    # the firt time train the model only on the verified files
    history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
    print("first train on only verified files")
    if(not continue_train):
        for epoch in range(first_phase_lp['e']):
            print("EPOCA: " + str(epoch) + " di " + str(first_phase_lp['e']))
            Xt = random_slice(X)
            Xt = np.expand_dims(Xt, axis=3)
            print(Xt.shape)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            h = model.fit(data=Xt, labels=Y, model_name=model_name, learning_rate=first_phase_lp['lr'], batch_size=first_phase_lp[
                'bs'], epochs=EPOCHS_PER_TRAIN_F, early_stop=early_stopping, validation_split=validation_split, verbose=v)
            del Xt
            history['acc'].append(h.history['acc'])
            history['loss'].append(h.history['loss'])
            history['val_acc'].append(h.history['val_acc'])
            history['val_loss'].append(h.history['val_loss'])
    del X
    del Y
    print("finish the firt train")
    print("*****************************************************************")
    while(files_used < pass_dataset * sum(classes_frequency.values())):
        # load the next num_for_each_train files
        X, Y, V = data_loader.get_next_spectrograms(
            spec_version, num_for_each_train)
        # obtain model statistics
        print("computing stat of the model...")
        output_vec, class_dict, how_many_per_class = model.compute_stat(X)
        if(verbose):
            print("ACTUAL MODEL STAT:")
            print(class_dict)
            print("--------------------------------------------------------------")
            print("how many files per class classified:")
            print(how_many_per_class)
            print()
        # discard some files
        print("self verify the unverified files...")
        indeces, loaded_per_class = self_verify(Y, V, output_vec, class_dict,
                                                self_verify_max_per_class, loaded_per_class, threshold)
        print("removed " + str(len(indeces)) + " files")
        X = np.delete(X, indeces, axis=0)
        Y = np.delete(Y, indeces, axis=0)
        print("dimension of the tensor for the training")
        print(X.shape)
        print("ready for the next train...")
        for epoch in range(second_phase_lp['e']):
            print("EPOCA: " + str(epoch) + " di " + str(second_phase_lp['e']))
            Xt = random_slice(X)
            Xt = np.expand_dims(Xt, axis=3)
            model.fit(data=Xt, labels=Y, model_name=model_name, learning_rate=second_phase_lp['lr'], batch_size=second_phase_lp[
                'bs'], epochs=EPOCHS_PER_TRAIN_S, early_stop=early_stopping, validation_split=validation_split, verbose=v)
            history['acc'].append(h.history['acc'])
            history['loss'].append(h.history['loss'])
            history['val_acc'].append(h.history['val_acc'])
            history['val_loss'].append(h.history['val_loss'])
        files_used += num_for_each_train
        del X
        del Y
        del V
        del indeces
        del output_vec
        del Xt

    print("finish the firt train")
    model.print_history(model_name=model_name, history=history)
    model.save_history(model_name=model_name, history=history)
    # save the model
    print("saving the model...")
    model.save_model(model_name=model_name)
    print("training finished")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the convolutional neural network ")
    parser.add_argument('--model_name', help='name of the model.')
    parser.add_argument(
        '--r', help='continue to train the model selected.', action='store_true')
    parser.add_argument(
        '--sr', help='sample rate for the audio signals', type=int, default=32000)
    parser.add_argument(
        '--sv', help='spectrogram version to compute (1 or 2).', type=int, default=1)
    parser.add_argument(
        '--dtc', help='dimension to conform the spectrograms', type=int, default=3000)
    parser.add_argument(
        '--nfa', help='number of frame to use to train', type=int, default=384)
    parser.add_argument(
        '--th', help='threshold for self verifying a file', type=int, default=5)
    parser.add_argument(
        '--nos', help='number of spectrograms for each train', type=int, default=3000)
    parser.add_argument(
        '--nfi', help='number of convolutional filters', type=int, default=48)
    parser.add_argument(
        '--flr', help='learning rate for the firt learning phase', type=float, default=0.0001)
    parser.add_argument(
        '--fe', help='ephocs for the firt learning phase', type=int, default=30)
    parser.add_argument(
        '--fbs', help='batch size for the firt learning phase', type=int, default=32)
    parser.add_argument(
        '--slr', help='learning rate for the second learning phase', type=float, default=0.0002)
    parser.add_argument(
        '--se', help='ephocs for the second learning phase', type=int, default=50)
    parser.add_argument(
        '--pd', help='how many times use the dataset files to train', type=int, default=2)
    parser.add_argument(
        '--sbs', help='batch size for the second learning phase', type=int, default=64)
    parser.add_argument(
        '--vs', help='validation split percentage', type=float, default=0.2)
    parser.add_argument(
        '--es', help='early_stopping the training', action='store_true')
    parser.add_argument(
        '--v', help='verbose', action='store_true')
    args = parser.parse_args()

    main(model_name=args.model_name, continue_train=args.r, sample_rate=args.sr, spec_version=args.sv, dimension_conform=args.dtc, num_of_frame=args.nfa, threshold=args.th, pass_dataset=args.pd,
         num_for_each_train=args.nos, num_of_filter=args.nfi, first_phase_lp={'lr': args.flr, 'e': args.fe, 'bs': args.fbs}, second_phase_lp={'lr': args.slr, 'e': args.se, 'bs': args.sbs}, validation_split=args.vs, early_stopping=args.es, verbose=args.v)
