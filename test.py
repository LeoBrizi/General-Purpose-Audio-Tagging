from DataLoader import *
from CNNSpecNetwork import *
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import argparse
import matplotlib.colors as colors
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.set_size_inches(18.5, 10.5)
    return ax


def main(model_name, spec_version=1):
    model = CNNSpecNetwork()
    model.load_model(model_name)
    loader = DataLoader()
    loader.load_files_labels("./dataset/test_post_competition.csv")
    diz_mapping = loader.get_label_id_mapping()
    print(diz_mapping)
    spectrograms, label = loader.load_verified_spectrograms(version=spec_version)
    print("Predict on test set...")
    y_pred = []
    for s in tqdm(spectrograms):
        y = model.predict_single_data(s)
        y_pred.append(y[0])

    y_test_arr = np.array(label)
    y_pred_arr = np.array(y_pred)

    confusionMatrix = confusion_matrix(y_test_arr, y_pred_arr)
    classes=[]
    for k in diz_mapping:
        classes.insert(diz_mapping[k], k)
    plot_confusion_matrix(y_test_arr, y_pred_arr, classes=classes,
                      title='Confusion matrix, without normalization')
    plt.show()
    print('Accuracy:')
    print(accuracy_score(y_test_arr, y_pred_arr))
    print('Precision:')
    print(precision_score(y_test_arr, y_pred_arr, average=None))
    print('Recall:')
    print(recall_score(y_test_arr, y_pred_arr, average=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the convolutional neural network ")
    parser.add_argument('--model_name', help='name of the model.')
    parser.add_argument(
        '--sv', help='spectrogram version used to train the network', type=int, default=1)
    args = parser.parse_args()

    main(model_name=args.model_name, spec_version=args.sv)
