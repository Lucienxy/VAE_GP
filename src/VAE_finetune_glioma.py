# An auto-enconder by pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from model import VAE


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Task2Loss(nn.Module):
    def __init__(self):
        super(Task2Loss, self).__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        return loss


def get_mean_r2(real_d, predict_d):
    r2_list = []
    for i in range(real_d.shape[0]):
        r2_list.append(r2_score(real_d[i,:], predict_d[i,:]))
    return np.mean(r2_list)



torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


df = pd.read_csv('./../example_data/glimas_pro_df.csv')

pid_df = df[['PID']]

embedding_dim = 64
embedding_df = pd.read_csv('./../example_data/gliomas_pro_emb_df.csv')
embedding_df.drop(['PID'],axis=1,inplace=True)
embedding_values = embedding_df.values

df.drop(['PID'],axis=1,inplace=True)
input_size = df.shape[1]
data_numpy = df.values

rd=10
hidden_size = embedding_dim
num_epochs = 1000
batch_size = 128
t2_w = 1
p_drop = 0.1
node_num = [512, 256, 128]
# node_num = [1024, 512, 128]

save_interval=1000
KLD_weight = 1
save_path = './../example_data/finetune/'
new_dir = 'Layer_'+str(node_num[0])+'_'+str(node_num[1])+'_'+str(node_num[2])+'_Drop_'+str(p_drop)+'_BS_'+str(batch_size) +'_RD_'+str(rd)
if not os.path.exists(save_path+new_dir):
    os.makedirs(save_path+new_dir)
mat_pth = save_path+new_dir

trained_dir = './../example_data/pretrain/Layer_512_256_128_Drop_0.1_BS_256_RD_100/BN_VAE_best.pth'
trained_df = './../example_data/merged_pro_overlap_df.csv'
train_input = trained_df.shape[0]-1
train_hidden = embedding_dim


x_train, x_test, y_train, y_test = train_test_split(data_numpy, embedding_values, test_size=0.1, random_state=rd)

train_indices = np.where(np.all(data_numpy == x_train[:, None], axis=2))[1]
train_pid_list = pid_df.iloc[train_indices,0].values.tolist()
test_indices = np.where(np.all(data_numpy == x_test[:, None], axis=2))[1]
test_pid_list = pid_df.iloc[test_indices,0].values.tolist()

x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

data_tensor = torch.tensor(data_numpy, dtype=torch.float32)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)  

loss_pth_train = save_path+new_dir+'/train_loss.csv'
loss_pth_test = save_path+new_dir+'/test_loss.csv'
# save train and test pid as csv
train_pid_df = pd.DataFrame(train_pid_list)
train_pid_df.to_csv(mat_pth+'/train_pid.csv',index=None)
test_pid_df = pd.DataFrame(test_pid_list)
test_pid_df.to_csv(mat_pth+'/test_pid.csv',index=None)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
vae = VAE(input_size=input_size, hidden_size=hidden_size,node_num=node_num,p_drop=p_drop)

trained_vae = VAE(input_size=train_input, hidden_size=train_hidden,node_num=node_num,p_drop=p_drop)
trained_vae.load_state_dict(torch.load(trained_dir))

# copy the weights
key_copy = ['fc1.weight', 'fc1.bias', 'fc8.weight', 'fc8.bias']
state_dict_m1 = trained_vae.state_dict()
state_dict_to_load = {k: v for k, v in state_dict_m1.items() if k not in key_copy}
vae_state_dict = vae.state_dict()
vae_state_dict.update(state_dict_to_load)
vae.load_state_dict(vae_state_dict, strict=False)

# freeze the weights
# for name, param in vae.named_parameters():
#     if name in ['fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'fc6.weight', 'fc6.bias', 'fc7.weight', 'fc7.bias']:
#         param.requires_grad = False

task2_loss_fn = Task2Loss()
optimizer = optim.Adam(vae.parameters(), lr=1e-4)


# train
losses_train = []
losses_test = []
r2_list = []
min_test_loss = 1
min_test_epoch = 0
min_test_res = []
min_train_loss = 1
min_train_epoch = 0

for epoch in range(num_epochs):
    optimizer.zero_grad()
    for batch_x, batch_y in train_dataloader:
        outputs, mu, logvar, z, z_mean = vae(batch_x)
        task1_loss, MSE_loss, KLD_loss = vae.loss_function(outputs, batch_x, mu, logvar, KLD_weight)
        task2_loss = task2_loss_fn(z_mean, batch_y)
        all_loss = task1_loss + task2_loss*t2_w
        all_loss.backward()
        optimizer.step()
    vae.eval()
    with torch.no_grad():
        outputs, mu, logvar, z,z_mean = vae(x_train)
        task1_loss, MSE_loss, KLD_loss = vae.loss_function(outputs, x_train, mu, logvar, KLD_weight)
        task2_loss = task2_loss_fn(z_mean, y_train)
        outputs_t, mu_t, logvar_t, z_t,z_mean_t = vae(x_test)
        task1_loss_t, MSE_loss_t, KLD_loss_t = vae.loss_function(outputs_t, x_test, mu_t, logvar_t, KLD_weight)
        task2_loss_t = task2_loss_fn(z_mean_t, y_test)
        train_t1_r2 = get_mean_r2(x_train.detach().numpy(), outputs.detach().numpy())
        train_t2_r2 = get_mean_r2(y_train.detach().numpy(), z_mean.detach().numpy())
        test_t1_r2 = get_mean_r2(x_test.detach().numpy(), outputs_t.detach().numpy())
        test_t2_r2 = get_mean_r2(y_test.detach().numpy(), z_mean_t.detach().numpy())
        losses_train.append([MSE_loss.item(),KLD_loss.item(),task1_loss.item(),task2_loss.item()])
        losses_test.append([MSE_loss_t.item(),KLD_loss_t.item(),task1_loss_t.item(),task2_loss_t.item()])
        r2_list.append([train_t1_r2,train_t2_r2,test_t1_r2,test_t2_r2])
        # print(f"Epoch: {epoch + 1}, MSE loss: {MSE_loss.item():.4f}, KLD loss: {KLD_loss.item():.4f}, Loss1: {task1_loss.item():.4f}, MSE loss_t: {MSE_loss_t.item():.4f}, KLD loss_t: {KLD_loss_t.item():.4f}, Loss1_t: {task1_loss_t.item():.4f}")
        s1 = f"Epoch: {epoch + 1}, MSE loss: {MSE_loss.item():.4f}, KLD loss: {KLD_loss.item():.4f}, Loss1: {task1_loss.item():.4f}, Loss2: {task2_loss.item():.4f}, MSE loss_t: {MSE_loss_t.item():.4f}, KLD loss_t: {KLD_loss_t.item():.4f}, Loss1_t: {task1_loss_t.item():.4f}, Loss2_t: {task2_loss_t.item():.4f}"
        s2 = f"Epoch: {epoch + 1}, train_t1_r2: {train_t1_r2:.4f}, train_t2_r2: {train_t2_r2:.4f}, test_t1_r2: {test_t1_r2:.4f}, test_t2_r2: {test_t2_r2:.4f}"
        print(s1)
        print(s2)
        if (epoch + 1) % save_interval == 0:
            torch.save(vae.state_dict(), mat_pth+'/vae_assign_'+str(hidden_size)+'_'+str(epoch+1)+'.pth')
            outputs_all, mu_all, logvar_all, z_all,z_mean_all = vae(data_tensor)
            mu_df = pd.DataFrame(z_mean_all.detach().numpy())
            mu_df = pid_df.join(mu_df)
            mu_df.to_csv(mat_pth+'/vae_assign_'+str(hidden_size)+'_'+str(epoch+1)+'_mu.csv',index=None)
            z_df = pd.DataFrame(z_all.detach().numpy())
            z_df = pid_df.join(z_df)
            z_df.to_csv(mat_pth+'/vae_assign_'+str(hidden_size)+'_'+str(epoch+1)+'_z.csv',index=None)
        if MSE_loss_t.item() < min_test_loss:
            min_test_loss = MSE_loss_t.item()
            min_test_epoch = epoch+1
            min_test_res = [s1,s2]
            torch.save(vae.state_dict(), mat_pth+'/BN_VAE_best.pth')
            outputs_all, mu_all, logvar_all, z_all,z_mean_all = vae(data_tensor)
            mu_df = pd.DataFrame(z_mean_all.detach().numpy())
            mu_df = pid_df.join(mu_df)
            mu_df.to_csv(mat_pth+'/BN_VAE_best_mu.csv',index=None)
            z_df = pd.DataFrame(z_all.detach().numpy())
            z_df = pid_df.join(z_df)
            z_df.to_csv(mat_pth+'/BN_VAE_best_z.csv',index=None)
    vae.train()
print("Encoded data shape:", mu.shape)
print('Best Epoch:',min_test_epoch)
print(min_test_res[0])
print(min_test_res[1])

# torch.save(vae.state_dict(), mat_pth+'/vae_assign_'+str(hidden_size)+'_'+str(epoch+1)+'.pth')

loss_df_train = pd.DataFrame(losses_train)
loss_df_train.to_csv(loss_pth_train,index=None)
loss_df_test = pd.DataFrame(losses_test)
loss_df_test.to_csv(loss_pth_test,index=None)
r2_df = pd.DataFrame(r2_list)
r2_df.to_csv(mat_pth+'/R2.csv',index=None)
