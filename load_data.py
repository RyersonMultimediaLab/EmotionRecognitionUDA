import numpy as np
from copy import deepcopy
import torch

class Load_Data():
    def load_P_DEAP(self, subjectList_DEAP):
      #Load DEAP pre-processed data
      DataSeg_PSD_all_trials_DEAP=torch.zeros((1,160)) #160=32 channels*5 frequency bands
      DataSeg_DE_all_trials_DEAP=torch.zeros((1,160))
      labels_all_trials_DEAP=torch.zeros((1,1))
      count=0
      data_P_path_DEAP='/data/DEAP/preprocessed/'

      for sub in subjectList_DEAP:
          PSD_tmp=np.load(data_P_path_DEAP+'PSD_sub_'+sub+'.npy')
          DE_tmp=np.load(data_P_path_DEAP+'DE_sub_'+sub+'.npy')
          labels_tmp=np.load(data_P_path_DEAP+'labels_sub_'+sub+'.npy')

          if count==0:
            DataSeg_PSD_all_trials_DEAP=PSD_tmp
            DataSeg_DE_all_trials_DEAP=DE_tmp
            labels_all_trials_DEAP=labels_tmp
          else:
            DataSeg_PSD_all_trials_DEAP=np.concatenate((DataSeg_PSD_all_trials_DEAP,PSD_tmp), axis=0)
            DataSeg_DE_all_trials_DEAP=np.concatenate((DataSeg_DE_all_trials_DEAP,DE_tmp), axis=0)
            labels_all_trials_DEAP=np.concatenate((labels_all_trials_DEAP,labels_tmp), axis=0)

          count=count+1

      #2 discrete categories: Positive and Negative
      labels_DEAP_dis_all = deepcopy(labels_all_trials_DEAP)
      for i in range(labels_DEAP_dis_all.shape[0]):
        if labels_all_trials_DEAP[i][0]<=4.5:
          labels_DEAP_dis_all[i][0]=0
        else:
          labels_DEAP_dis_all[i][0]=1

      return DataSeg_PSD_all_trials_DEAP,DataSeg_DE_all_trials_DEAP,labels_DEAP_dis_all


    def load_P_SEED(self, subjectList_SEED):
      #Load SEED pre-processed data
      DataSeg_PSD_all_trials_SEED=torch.zeros((1,160)) #160=32 channels*5 frequency bands
      DataSeg_DE_all_trials_SEED=torch.zeros((1,160))
      labels_all_trials_SEED=torch.zeros((1,1))
      count=0
      data_P_path_SEED='/data/SEED/preprocessed/'

      for sub in subjectList_SEED:
          PSD_tmp=np.load(data_P_path_SEED+'PSD_sub_'+sub+'.npy')
          DE_tmp=np.load(data_P_path_SEED+'DE_sub_'+sub+'.npy')
          labels_tmp=np.load(data_P_path_SEED+'labels_sub_'+sub+'.npy')

          if count==0:
            DataSeg_PSD_all_trials_SEED=PSD_tmp
            DataSeg_DE_all_trials_SEED=DE_tmp
            labels_all_trials_SEED=labels_tmp
          else:
            DataSeg_PSD_all_trials_SEED=np.concatenate((DataSeg_PSD_all_trials_SEED,PSD_tmp), axis=0)
            DataSeg_DE_all_trials_SEED=np.concatenate((DataSeg_DE_all_trials_SEED,DE_tmp), axis=0)
            labels_all_trials_SEED=np.concatenate((labels_all_trials_SEED,labels_tmp), axis=0)

          count=count+1

      #Take samples with labels 0 and 1
      condition1 = labels_all_trials_SEED == 0
      condition2 = labels_all_trials_SEED == 2
      indices = np.where(np.logical_or(condition1, condition2))
      labels_all_trials_SEED=labels_all_trials_SEED[indices[0],:]
      DataSeg_PSD_all_trials_SEED=DataSeg_PSD_all_trials_SEED[indices[0],:]
      DataSeg_DE_all_trials_SEED=DataSeg_DE_all_trials_SEED[indices[0],:]

      return DataSeg_PSD_all_trials_SEED,DataSeg_DE_all_trials_SEED,labels_all_trials_SEED
