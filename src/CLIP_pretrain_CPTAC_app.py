import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import StandardScaler
from model import EmbeddingModel

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

node_num = [1024,512,256,256,128]
embedding_dim = 64
learning_rate = 1E-3
batch_size = 64
num_epochs = 197
rd = 0
save_pth = './../example_data/embedding/pretrain/'
cohort = '_CPTAC'



pro_dir = './../example/merged_pro_overlap_df.csv'
pro_df = pd.read_csv(pro_dir)
pid_df = pro_df[['PID']]
pro_df.drop(['PID'],axis=1,inplace=True)
pro_data = torch.tensor(pro_df.values, dtype=torch.float32)

sample_num = pro_data.shape[0]

pro_encoder = EmbeddingModel(pro_data.shape[1],node_num,embedding_dim)
# pro_pretrain_pth = save_pth+f'/pro_encoder_{num_epochs}.pth'
pro_pretrain_pth = save_pth+f'/pro_encoder_best.pth'
pro_encoder.load_state_dict(torch.load(pro_pretrain_pth))
pro_emb = pro_encoder(pro_data)
pro_emb = pro_emb.detach().numpy()
pro_emb = StandardScaler().fit_transform(pro_emb.T).T
pro_emb_df = pd.DataFrame(pro_emb)
pro_emb_df.insert(0,'PID',pid_df['PID'])
pro_emb_df.to_csv(save_pth+'/pro_emb_df'+cohort+'.csv',index=None)


rna_dir = './../example/merged_rna_overlap_df.csv'
rna_df = pd.read_csv(rna_dir)
pid_df = rna_df[['PID']]
rna_df.drop(['PID'],axis=1,inplace=True)
rna_data = torch.tensor(rna_df.values, dtype=torch.float32)

sample_num = rna_data.shape[0]

rna_encoder = EmbeddingModel(rna_data.shape[1],node_num,embedding_dim)
# rna_pretrain_pth = save_pth+f'/rna_encoder_{num_epochs}.pth'
rna_pretrain_pth = save_pth+f'/rna_encoder_best.pth'
rna_encoder.load_state_dict(torch.load(rna_pretrain_pth))
rna_emb = rna_encoder(rna_data)
rna_emb = rna_emb.detach().numpy()
rna_emb = StandardScaler().fit_transform(rna_emb.T).T
rna_emb_df = pd.DataFrame(rna_emb)
rna_emb_df.insert(0,'PID',pid_df['PID'])
rna_emb_df.to_csv(save_pth+'/rna_emb_df'+cohort+'.csv',index=None)
