# General Purpose Audio Tagging

Neural network project addresses the problem of general-purpose automatic audio tagging. (http://dcase.community/challenge2018/task-general-purpose-audio-tagging). This project uses a convolutional neural network (VGG) to classify 41 classes of audio.
Dowload the dataset [here](https://www.kaggle.com/c/freesound-audio-tagging/data).
To initialize the workspace use the init_work_space.sh bash script.
If you want to train a model:
```
python3 train_spec.py --model_name "name of the new model"
```
If you want to try a pretrained model, first lunch the `init_work_space.sh` script with the `-m` option and then:
```
python3 test.py --model_name test9
```
