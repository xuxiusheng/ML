from scipy import fft
from scipy.io import wavfile
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from pprint import pprint

def load_data(path):
    labels = os.listdir(path)
    total = 0
    train_path = 'trainSet'
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    for label in labels:
        class_path = os.path.join(path, label, 'converted')
        total += len(os.listdir(class_path))
    features = np.empty((total, 1000))
    y = np.array([])
    i = 0
    for label in labels:
        class_path = os.path.join(path, label, 'converted')
        waves = os.listdir(class_path)
        for wave in waves:
            file_path = os.path.join(class_path, wave)
            (Sample_rate, X) = wavfile.read(file_path)
            F = abs(fft(X)[:1000])
            # name = wave[:-4] + '.fft'
            # save_path = os.path.join(train_path, name)
            # np.save(save_path, F)
            features[i] = F
            i += 1
            y = np.append(y, labels.index(label))
    print(features.shape)
    print(y.shape)
    return features, y

def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(multi_class='multinomial', max_iter=3000)
    model = model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    model_output = open('model.pkl', 'wb')
    pickle.dump(model, model_output)
    model_output.close()


def testModel(X, y):
    pkl_file = open('model.pkl', 'rb')
    model = pickle.load(pkl_file)
    pprint(model)
    pkl_file.close()


X, y = load_data('.\genres')
# model(X, y)
testModel(X, y)