import numpy as np
import matplotlib.pyplot as plt
import pickle

test_path = 'E:/TF/ShareWeight/cifar10/cifar10_data/cifar-10-batches-py/test_batch'    
with open(test_path, 'rb') as f:
     datadict = pickle.load(f,encoding='latin1')
     
X = datadict["data"] 
Y = datadict['labels']
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

#Visualizing CIFAR 10
fig, axes1 = plt.subplots(5,5,figsize=(3,3))
for j in range(5):
    for k in range(5):
        i = np.random.choice(range(len(X)))
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(X[i:i+1][0])
