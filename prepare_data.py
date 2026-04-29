import numpy as np
import random
import torch
from sklearn.utils import shuffle

class Prepare_Data():
  def create_mini_batches(self, data, labels, batch_size):
    data_size=data.shape[0]
    itr_len=int(data_size/batch_size)
    data=data[0:itr_len*batch_size]
    labels=labels[0:itr_len*batch_size]
    #Create batches
    data_batch= np.empty([itr_len, batch_size, data.shape[1], data.shape[2]])
    label_batch=np.empty([itr_len, batch_size])
    j=0

    for i in range(itr_len):
      data_batch[i]=data[j:j+batch_size]
      label_batch[i]=labels[j:j+batch_size]
      j=j+batch_size

    label_batch=label_batch.astype(int)

    return data_batch, label_batch

  def Prepare_DEAP(self, DataSeg_PSD_all_trials_DEAP,DataSeg_DE_all_trials_DEAP,labels_all_trials_DEAP,batch_size,num_classes):
    #Normalization [-1,1]
    tmp=np.zeros((DataSeg_DE_all_trials_DEAP.shape[0],DataSeg_DE_all_trials_DEAP.shape[1]))
    for i in range (DataSeg_DE_all_trials_DEAP.shape[0]):
      tmp[i]=(((DataSeg_DE_all_trials_DEAP[i]-np.amin(DataSeg_DE_all_trials_DEAP[i]))/(np.amax(DataSeg_DE_all_trials_DEAP[i])-np.amin(DataSeg_DE_all_trials_DEAP[i])))*2)-1

    DataSeg_DE_all_trials_DEAP=tmp

    tmp=np.zeros((DataSeg_PSD_all_trials_DEAP.shape[0],DataSeg_PSD_all_trials_DEAP.shape[1]))
    for i in range (DataSeg_PSD_all_trials_DEAP.shape[0]):
      tmp[i]=(((DataSeg_PSD_all_trials_DEAP[i]-np.amin(DataSeg_PSD_all_trials_DEAP[i]))/(np.amax(DataSeg_PSD_all_trials_DEAP[i])-np.amin(DataSeg_PSD_all_trials_DEAP[i])))*2)-1

    DataSeg_PSD_all_trials_DEAP=tmp
    random_seed = 8
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #Reshape
    Data_DE_DEAP=np.reshape(DataSeg_DE_all_trials_DEAP,(DataSeg_DE_all_trials_DEAP.shape[0],DataSeg_DE_all_trials_DEAP.shape[1],1))
    Data_PSD_DEAP=np.reshape(DataSeg_PSD_all_trials_DEAP,(DataSeg_PSD_all_trials_DEAP.shape[0],DataSeg_PSD_all_trials_DEAP.shape[1],1))
    labels_DEAP = labels_all_trials_DEAP.ravel()
    inp_shape=Data_DE_DEAP.shape[1]
    Data_DE_DEAP, Data_PSD_DEAP, labels_DEAP = shuffle(Data_DE_DEAP, Data_PSD_DEAP, labels_DEAP, random_state=random_seed)
    data_batch_DEAP, label_batch_DEAP = self.create_mini_batches(Data_PSD_DEAP, labels_DEAP, batch_size)

    return data_batch_DEAP,label_batch_DEAP

  def Prepare_SEED(self, DataSeg_PSD_all_trials_SEED, DataSeg_DE_all_trials_SEED, labels_all_trials_SEED,batch_size,num_classes):
     #Normalization [-1,1]
    tmp=np.zeros((DataSeg_DE_all_trials_SEED.shape[0],DataSeg_DE_all_trials_SEED.shape[1]))
    for i in range (DataSeg_DE_all_trials_SEED.shape[0]):
      tmp[i]=(((DataSeg_DE_all_trials_SEED[i]-np.amin(DataSeg_DE_all_trials_SEED[i]))/(np.amax(DataSeg_DE_all_trials_SEED[i])-np.amin(DataSeg_DE_all_trials_SEED[i])))*2)-1

    DataSeg_DE_all_trials_SEED=tmp

    tmp=np.zeros((DataSeg_PSD_all_trials_SEED.shape[0],DataSeg_PSD_all_trials_SEED.shape[1]))
    for i in range (DataSeg_PSD_all_trials_SEED.shape[0]):
      tmp[i]=(((DataSeg_PSD_all_trials_SEED[i]-np.amin(DataSeg_PSD_all_trials_SEED[i]))/(np.amax(DataSeg_PSD_all_trials_SEED[i])-np.amin(DataSeg_PSD_all_trials_SEED[i])))*2)-1

    DataSeg_PSD_all_trials_SEED=tmp
    random_seed = 8
    random.seed(random_seed)
    #Reshape
    Data_DE_SEED=np.reshape(DataSeg_DE_all_trials_SEED,(DataSeg_DE_all_trials_SEED.shape[0],DataSeg_DE_all_trials_SEED.shape[1],1))
    Data_PSD_SEED=np.reshape(DataSeg_PSD_all_trials_SEED,(DataSeg_PSD_all_trials_SEED.shape[0],DataSeg_PSD_all_trials_SEED.shape[1],1))
    labels_SEED = labels_all_trials_SEED.ravel()
    inp_shape=Data_DE_SEED.shape[1]
    Data_DE_SEED, Data_PSD_SEED, labels_SEED = shuffle(Data_DE_SEED, Data_PSD_SEED, labels_SEED, random_state=random_seed)
    data_batch_SEED, label_batch_SEED= self.create_mini_batches(Data_PSD_SEED, labels_SEED, batch_size)

    return data_batch_SEED,label_batch_SEED
