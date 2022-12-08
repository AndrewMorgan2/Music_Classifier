#Let's get the data in
from dataset import GTZAN
import librosa
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc 
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

data_augment = []
dataset_normal = GTZAN('../data/train.pkl')
##Because I'm changing the training set I have to do the same to the validation set

data_normal = dataset_normal.dataset

##This next part breaks the task into 4 so that I can run the program 4 times (I dont have the memory to look at all the data at once)
quater = 3
add_noise = True

sep = None
sep_top = None
##Quater the dataset
if quater == 0:
    sep = 0
    sep_top = round((len(data_normal)/4))
if quater == 1:
    sep = round((len(data_normal)/4)) + 1
    sep_top = round((len(data_normal)/2))
if quater == 2:
    sep = round((len(data_normal)/2)) + 1
    sep_top = round((len(data_normal)/4) * 3)
if quater == 3:
    sep = round((len(data_normal)/4)  * 3)  + 1
    sep_top = len(data_normal)

print(sep, " to ", sep_top)
data_for_run = data_normal[sep:sep_top]

print("Starting...")
print("Augmenting training data...")
index = 0
for data_point in data_for_run:
    if index == len(data_for_run)/4:
        print("25%...")
        gc.collect()
    if index == (len(data_for_run)/4) * 2:
        print("50%...")
        gc.collect()
    if index == (len(data_for_run)/4) * 3:
        print("75%...")
        gc.collect()
    
    print(index, "/", sep_top)
    index += 1
    sound_track = data_point[3]
    length_clip = sound_track.shape[0]

    if add_noise:
        #Added noise (Guassian)
        RMS=np.sqrt(np.mean(sound_track)**2)

        noise=np.random.normal(0, RMS, length_clip)
        noise_big = noise *150
        noise_bigger = noise *300 #Too much 

        #Makes 3 new data points and loads in the old one 
        noises = [noise, noise_big, noise_bigger]

        ##make 0,4 to get noise_bigger
        for noise_amount in range(0, 3):
            if noise_amount == 0:
                noisy_sound_track = sound_track
            else:
                noisy_sound_track = sound_track + noises[noise_amount - 1]
            #Remake the spectogram
            melspectogram = librosa.feature.melspectrogram(y=noisy_sound_track, win_length = 1024, hop_length = 512, n_mels= 80)
            melspectogram_power = np.log(melspectogram + 1e-6)
            melspectogram_power = np.pad(melspectogram_power, pad_width=((0,0), (0,39)), mode = 'edge')
            melspectogram_power_tensor = torch.tensor(np.array([melspectogram_power])).float()

            #Putting back in dataset in tuple form 
            tuple_melspectogram = (data_point[0], melspectogram_power_tensor, data_point[2], data_point[3])
            data_augment.append(tuple_melspectogram)
    else:
        #Reducing noise

        #Makes 3 new data points and loads in the old one 
        kernel_sizes = [10, 15, 20]

        ##make 0,4 to get noise_bigger
        for reduce_amount in range(0, 4):
            if reduce_amount == 0:
                new_sound_track = sound_track
            else:
                kernel = np.ones(kernel_sizes[reduce_amount - 1]) / kernel_sizes[reduce_amount - 1]
                new_sound_track = np.convolve(sound_track, kernel, mode='same')

            #Remake the spectogram
            melspectogram = librosa.feature.melspectrogram(y=new_sound_track, win_length = 1024, hop_length = 512, n_mels= 80)
            melspectogram_power = np.log(melspectogram + 1e-6)
            melspectogram_power = np.pad(melspectogram_power, pad_width=((0,0), (0,39)), mode = 'edge')
            melspectogram_power_tensor = torch.tensor(np.array([melspectogram_power])).float()

            #Putting back in dataset in tuple form 
            tuple_melspectogram = (data_point[0], melspectogram_power_tensor, data_point[2], data_point[3])
            data_augment.append(tuple_melspectogram)

print("let's check that dimensionality ", len(data_augment), " with ", len(data_augment[0]), " per item")

print("Putting it all back into pkl...")
destination = ""
if add_noise:
    destination = "./noise_train_" + str(quater) + ".pkl"
else:
    destination = "./reduced_train_" + str(quater) + ".pkl"

##Use pickle to put it into .pkl
with open(destination,'wb') as f:
    pickle.dump(data_augment, f)

print("test_new.pkl done")
