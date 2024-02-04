# An auto-enconder by pytorch

import torch
import pandas as pd
import numpy as np
import random
import os
from model import VAE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)



def gen_data_from_rdz(mat_pth, num_samples=100):
    with torch.no_grad():
        z = torch.randn(num_samples, hidden_size)
        generated_data = vae.decode(z)
    generated_df = pd.DataFrame(generated_data.numpy())
    # z_df = pd.DataFrame(z.numpy())
    generated_df.to_csv(mat_pth+'/generated_data_'+str(num_samples)+'.csv',index=False)


def gene_data_TTS(mat_pth, train_pid, test_pid):
    z = pd.read_csv(mat_pth+'/BN_VAE_best_mu.csv')
    z_train = z[z['PID'].isin(train_pid)]
    z_test = z[z['PID'].isin(test_pid)]
    num_samples = z_train.shape[0]
    z_train.drop(['PID'],axis=1,inplace=True)
    z_train = torch.tensor(z_train.values, dtype=torch.float32)
    with torch.no_grad():
        generated_data = vae.decode(z_train)
    generated_df = pd.DataFrame(generated_data.numpy())
    generated_df.to_csv(mat_pth+'/generated_data_'+str(num_samples)+'.csv',index=False)
    num_samples = z_test.shape[0]
    z_test.drop(['PID'],axis=1,inplace=True)
    z_test = torch.tensor(z_test.values, dtype=torch.float32)
    with torch.no_grad():
        generated_data = vae.decode(z_test)
    generated_df = pd.DataFrame(generated_data.numpy())
    generated_df.to_csv(mat_pth+'/generated_data_'+str(num_samples)+'.csv',index=False)


def gen_data_new_cohort(mat_pth,z_path):
    z = pd.read_csv(z_path)
    num_samples = z.shape[0]
    z.drop(['PID'],axis=1,inplace=True)
    z = torch.tensor(z.values, dtype=torch.float32)
    with torch.no_grad():
        generated_data = vae.decode(z)
    generated_df = pd.DataFrame(generated_data.numpy())
    generated_df.to_csv(mat_pth+'/generated_data_'+str(num_samples)+'.csv',index=False)


save_path = './../example_data/finetune/'
train_df = pd.read_csv('./../example_data/glimas_pro_df.csv')
input_size =  train_df.shape[1]-1


hidden_size = 64
batch_size = 256
t2_w = 1
p_drop = 0.1
node_num = [512,256,128]
save_interval=1000
KLD_weight = 1
rd = 10

new_dir = 'Layer_'+str(node_num[0])+'_'+str(node_num[1])+'_'+str(node_num[2])+'_Drop_'+str(p_drop)+'_BS_'+str(batch_size)+'_RD_'+str(rd)
if not os.path.exists(save_path+new_dir):
    os.makedirs(save_path+new_dir)
mat_pth = save_path+new_dir

model_path = mat_pth+'/BN_VAE_best.pth'
vae = VAE(input_size=input_size, hidden_size=hidden_size,node_num=node_num,p_drop=p_drop)

vae.load_state_dict(torch.load(model_path))

gen_data_from_rdz(mat_pth, num_samples=100)
gen_data_from_rdz(mat_pth, num_samples=1000)
train_df = pd.read_csv(mat_pth+'/train_pid.csv')
test_df = pd.read_csv(mat_pth+'/test_pid.csv')
train_pid = train_df.iloc[:,0].values.tolist()
test_pid = test_df.iloc[:,0].values.tolist()
gene_data_TTS(mat_pth, train_pid, test_pid)



z_path = './../example_data/embedding/finetune/rna_emb_df_test.csv'
gen_data_new_cohort(mat_pth,z_path)


