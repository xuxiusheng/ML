import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from scipy.io import wavfile
import python_speech_features as mfcc
from sklearn import preprocessing
import os
import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def get_MFCC(sr, audio):
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy=False)
    feat = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i, :]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat
    features = preprocessing.scale(features)
    return features

source = './test_data/AudioSet/'
model_path = './'

gmm_files = [os.path.join(model_path, i) for i in os.listdir(model_path) if i[-3:] == 'GMM']
models = [pickle.load(open(i, 'rb')) for i in gmm_files]
test_files = [[os.path.join(source, 'male_clips', i) for i in os.listdir(os.path.join(source, 'male_clips'))],
              [os.path.join(source, 'female_clips', i) for i in os.listdir(os.path.join(source, 'female_clips'))]]

genders = ['male', 'female']
ind = 0
male_correct, female_correct = 0, 0
for content in test_files:
    for idx, file in enumerate(tqdm.tqdm(content)):
        sample_rate, audio = wavfile.read(file)
        features = get_MFCC(sample_rate, audio)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            model = models[i]
            scores = np.array(model.score(features))
            log_likelihood[i] = scores.sum()
        sex = np.argmax(log_likelihood)
        if ind == 0:
            if genders[sex] == 'male':
                male_correct += 1
            else:
                if genders[sex] == 'female':
                    female_correct += 1

        time.sleep(0.01)

    if ind == 0:
        male_rate = round(male_correct / len(content), 3)
    else:
        female_rate = round(female_correct / len(content), 3)
    ind += 1

print(male_rate, female_rate)
