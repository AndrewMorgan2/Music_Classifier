import os
import pickle
import gc
import dataset

merge = []

folder='./'
for filename in os.listdir(folder):  
    if filename.endswith('.pkl'):
        gc.collect()
        dataset_train = dataset.GTZAN(folder + filename)
        for data_point in dataset_train:
            merge.append(data_point)
        print(filename)
gc.collect()

myfile = open("train_reduce_noise.pkl","wb")
pickle.dump(merge, myfile)
myfile.close()