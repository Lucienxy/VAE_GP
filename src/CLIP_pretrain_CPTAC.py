import pandas as pd
import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import EmbeddingModel


def create_neg_indice_pair(num_samples):
    tmp_idx1 = torch.randperm(num_samples)
    tmp_idx2 = torch.randperm(num_samples)
    for i in range(num_samples - 1):
        if tmp_idx1[i] == tmp_idx2[i]:
            tmp_r = tmp_idx2[i].clone()
            tmp_idx2[i] = tmp_idx2[i + 1]
            tmp_idx2[i + 1] = tmp_r

    if tmp_idx1[-1] == tmp_idx2[-1]:
        tmp_r = tmp_idx2[-1].clone()
        tmp_idx2[-1] = tmp_idx2[0]
        tmp_idx2[0] = tmp_r
    return tmp_idx1, tmp_idx2


def create_neg_samples(embeddings, indices):
    negative_samples = embeddings[indices]
    return negative_samples


def compute_loss(batch_size, rna_embeddings, pro_embeddings):
    labels_pos = torch.ones(batch_size)
    # labels_neg = torch.zeros(batch_size)
    labels_neg = torch.ones(batch_size) * (-1)
    loss_pos = criterion(rna_embeddings, pro_embeddings, labels_pos)
    neg_idx_rna, neg_idx_pro = create_neg_indice_pair(batch_size)
    neg_rna_sample = create_neg_samples(rna_embeddings, neg_idx_rna)
    neg_pro_sample = create_neg_samples(pro_embeddings, neg_idx_pro)
    loss_neg = criterion(neg_rna_sample, neg_pro_sample, labels_neg)

    # loss = (loss_pos + loss_neg)
    # print('Pos Loss: {:.8f}, Neg Loss: {:.8f}'.format(loss_pos.item(), loss_neg.item()))
    return loss_pos, loss_neg


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

rna_dir = './../example_data/merged_rna_overlap_df.csv'
pro_dir = './../example_data/merged_pro_overlap_df.csv'


rna_df = pd.read_csv(rna_dir)
pro_df = pd.read_csv(pro_dir)
pid_df = rna_df[['PID']]
rna_df.drop(['PID'],axis=1,inplace=True)
pro_df.drop(['PID'],axis=1,inplace=True)
rna_data = torch.tensor(rna_df.values, dtype=torch.float32)
pro_data = torch.tensor(pro_df.values, dtype=torch.float32)


sample_num = rna_data.shape[0]

node_num = [1024,512,256,256,128]
embedding_dim = 64
learning_rate = 1E-4
batch_size = 32
num_epochs = 5
save_pth = './../example_data/embedding/pretrain/'
md_dir = save_pth+'/saved_model'

if not os.path.exists(md_dir):
    os.makedirs(md_dir)


rna_encoder = EmbeddingModel(rna_data.shape[1],node_num,embedding_dim)
pro_encoder = EmbeddingModel(pro_data.shape[1],node_num,embedding_dim)
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(list(rna_encoder.parameters()) + list(pro_encoder.parameters()), lr=learning_rate)


data_set = TensorDataset(rna_data, pro_data)
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
loss_list = []
min_loss = 10
min_epoch = 0
for epoch in range(num_epochs):
    for bat_rna, bat_pro in data_loader:
        optimizer.zero_grad()
        rna_embeddings = rna_encoder(bat_rna)
        pro_embeddings = pro_encoder(bat_pro)
        loss_pos,loss_neg = compute_loss(bat_rna.shape[0], rna_embeddings, pro_embeddings)
        loss = (loss_pos + loss_neg)
        loss.backward()
        optimizer.step()
    rna_encoder.eval()
    pro_encoder.eval()
    with torch.no_grad():
        rna_embeddings = rna_encoder(rna_data)
        pro_embeddings = pro_encoder(pro_data)
        loss_pos,loss_neg = compute_loss(rna_data.shape[0], rna_embeddings, pro_embeddings)
        loss = (loss_pos + loss_neg)
        loss_list.append([loss_pos.item(),loss_neg.item(),loss.item()])
        print(f'Epoch:{epoch+1}/{num_epochs}, pos:{loss_pos.item():.4f}, neg:{loss_neg.item():.4f},total:{loss.item():.4f}')
        if loss.item() < min_loss:
            min_loss = loss.item()
            min_epoch = epoch+1
            torch.save(rna_encoder.state_dict(), save_pth+f'/rna_encoder_best.pth')
            torch.save(pro_encoder.state_dict(), save_pth+f'/pro_encoder_best.pth')
    rna_encoder.train()
    pro_encoder.train()
    torch.save(rna_encoder.state_dict(), save_pth+f'saved_model/rna_encoder_{epoch+1}.pth')
    torch.save(pro_encoder.state_dict(), save_pth+f'saved_model/pro_encoder_{epoch+1}.pth')
loss_df = pd.DataFrame(loss_list,columns=['pos','neg','total_loss'])
loss_df.to_csv(save_pth+'/loss.csv',index=None)
print('Min loss',min_loss,'Min epoch',min_epoch)
