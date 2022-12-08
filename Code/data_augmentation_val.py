#Let's get the data in
from dataset import GTZAN
import librosa
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

data_validation_new = []
dataset_validation = GTZAN('../data/val.pkl')
data_validation = dataset_validation.dataset

print("Now making validation set...")
#Making a new validation set
for data_point in data_validation:
    #Remake the spectogram
    melspectogram = librosa.feature.melspectrogram(y=data_point[3], win_length = 1024, hop_length = 512, n_mels= 80)
    melspectogram_power = np.log(melspectogram + 1e-6)
    melspectogram_power = np.pad(melspectogram_power, pad_width=((0,0), (0,39)), mode = 'edge')
    melspectogram_power_tensor = torch.tensor(np.array([melspectogram_power])).float()

    #Putting back in dataset in tuple form 
    tuple_melspectogram = (data_point[0], melspectogram_power_tensor, data_point[2], data_point[3])
    data_validation_new.append(tuple_melspectogram)

with open('val_new.pkl','wb') as f:
    pickle.dump(data_validation_new, f)

print("val_new.pkl done")