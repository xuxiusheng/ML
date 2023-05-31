import os
import pickle
import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

def get_MFCC(sr, audio):
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy=False)
    features = preprocessing.scale(features)#对行做归一化，StandScaler是对列做归一化
    return features

source = 'train_data\youtube\male'

files = [os.path.join(source, i) for i in os.listdir(source)]

features = np.array([])
for f in files:
    sr, audio = wavfile.read(f)
    vector = get_MFCC(sr, audio)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
gmm = GaussianMixture(n_components=8, max_iter=200, covariance_type='diag', n_init=3).fit(features)

save_name = 'male.GMM'
with open(save_name, 'wb') as f:
    pickle.dump(gmm, f)
print('模型已保存')


 

