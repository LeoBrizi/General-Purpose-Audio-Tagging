import matplotlib.pyplot as plt 
import numpy as np
from CNNSpecNetwork import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import argparse
from tqdm import tqdm
import matplotlib.colors as colors
from Loader import *

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def createMatrix(rowCount, colCount, dataList):   
    mat = []
    for i in range (rowCount):
        rowList = []
        for j in range (colCount):
            if dataList[j] not in mat:
                rowList.append(dataList[j])
        mat.append(rowList)

    return mat 


def main(model_name, spec_version=1):
    #data = [[1, 2, 3], [4, 5, 6]]

    #x_header = data[0][1:]
    #y_header = [i for i in range(1, 13)]
    #data=data[1:]
    #for i in range(len(data)):
        #data[i] = data[i][1:]
    #arr = np.array(data)
    #fig, ax = plt.subplots()
    #norm = MidpointNormalize(midpoint=0)
    #im = ax.imshow(data, norm=norm, cmap=plt.cm.seismic, interpolation='none')

    #ax.set_xticks(np.arange(arr.shape[1]), minor=False)
    #ax.set_yticks(np.arange(arr.shape[0]), minor=False)
    #ax.xaxis.tick_top()
    #ax.set_xticklabels(x_header, rotation=90)
    #ax.set_yticklabels(y_header)

    #fig.colorbar(im)
    #plt.show()

    loader= Loader()
    loader.load_files_labels('./dataset/train.csv')
    res=Loader.get_general_statistics(loader)
    #return self.classes_frequency, classes_percent, verif_num, verif_num / tot, self.classes_verified, classes_percent_verified
    res_one=res[0]                                                                          #res1

    #k=list(res_one.keys())
    v=list(res_one.values())
    k=[]
    for i in range(1,42):
        k.append(i)
    
    x = np.random.normal(size = 3000)
    plt.hist(x, density=True, bins=30)
    x = np.arange(len(k))
    plt.bar(x, height=v)
    plt.xticks(x, k)
    plt.xticks(x, k, rotation='vertical')
    plt.show()

    res_two=res[1]                                                                          #res2
    labels = list(res_two.keys())
    sizes = list(res_two.values())
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=k,shadow=True)
    ax1.axis('equal')
    plt.show()

    res_three=res[2]                                                                        #res3   (number)
    print("\n")
    print("////////////////////////////////////")
    print("verified->number: ")
    print(res_three)
    print("////////////////////////////////////")

    res_four=res[3]                                                                         #res4   (number)
    print("\n")
    print("////////////////////////////////////")
    print("% verified->number: ")
    print(res_four)
    print("////////////////////////////////////")


    res_five=res[4]                                                                          #res5

    #k1=list(res_five.keys())
    k1=[]
    for i in range(1,42):
        k1.append(i)
    v1=list(res_five.values())
    x = np.random.normal(size = 3000)
    plt.hist(x, density=True, bins=30)
    x = np.arange(len(k1))
    plt.bar(x, height=v1)
    plt.xticks(x, k1)
    plt.xticks(x, k1, rotation='vertical')
    plt.show()



    res_six=res[5]
    #print(res_six)

    labels1 = list(res_six.keys())
    #print("?????????")
    #print(labels1)
    #print("?????????")
    sizes1 = list(res_six.values())
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes1, labels=k1,shadow=True)
    ax1.axis('equal')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the convolutional neural network ")
    parser.add_argument('--model_name', help='name of the model.')
    parser.add_argument(
        '--sv', help='spectrogram version used to train the network', type=int, default=1)
    args = parser.parse_args()

    main(model_name=args.model_name, spec_version=args.sv)
