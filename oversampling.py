'''
stable: unstable = 530000:530000
'''
import os
import torch
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
import random
from random import shuffle

seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)

global train_size
global test_size
global BATCH_SIZE
global train_unstable_size


BATCH_SIZE = 2560
EPOCHS = 100

writer = SummaryWriter('logs_780000')
data = np.load('city1_data.npy', allow_pickle=True)


train_size = int(len(data)*0.7)
test_size = len(data) - train_size
shuffle(data)
train_index = np.random.choice(len(data), train_size, replace=False)
train_dataset = data[:train_size]
test_dataset = data[train_size:]
train_label = train_dataset[:, -1]

index_unstable = np.where(train_label==1)[0]
index_stable = np.where(train_label==0)[0]


train_stable = train_dataset[index_stable]
train_unstable = train_dataset[index_unstable]
train_unstable = np.repeat(train_unstable, int(len(train_stable)/len(train_unstable)), axis=0)
train_unstable_size = train_unstable.shape[0]

train_stable_loader = torch.utils.data.DataLoader(train_stable, batch_size=BATCH_SIZE, shuffle=True)
train_unstable_loader = torch.utils.data.DataLoader(train_unstable, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Classifier(torch.nn.Module):
    def __init__(self, obs_dim, n_output):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(obs_dim - 2 + 64 * 2, 256)
        self.l2 = nn.Linear(256, 256)        
        self.day_embedding = nn.Embedding(7, 64)
        self.hour_embedding = nn.Embedding(24, 64)
        self.pre = nn.Linear(256, n_output)

    def forward(self, obs):
        dense_input = obs[:, :-2]
        day = obs[:, -2].long()
        hour = obs[:, -1].long()
        day_emb = self.day_embedding(day)
        hour_emb = self.hour_embedding(hour)
        x = torch.cat((dense_input, day_emb, hour_emb), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        pre = self.pre(x)        
        return pre

def train(model, train_stable_loader ,train_unstable_loader , optimizer, epoch):
    model.train()
    num_batch = 0   
    train_loss_sum = 0 
    for train_stable, train_unstable in zip(train_stable_loader, train_unstable_loader):  
        data_batch = torch.cat((train_stable, train_unstable), 0)
        train_x = data_batch[:, :-1].to(torch.float32)
        train_y = data_batch[:, -1].to(torch.long)
        optimizer.zero_grad()               
        output = model(train_x)                
        train_loss = loss_func(output, train_y)    
        train_loss_sum += train_loss.detach().item()        
        train_loss.backward()  
        optimizer.step()
        if num_batch%30 == 0: 
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, int(num_batch * BATCH_SIZE), train_unstable_size,
                train_loss.item()))    
        num_batch+=1   
    writer.add_scalar('train_loss',train_loss_sum/num_batch, epoch)  

    

def test(model, test_loader, epoch):
    test_loss = 0                           
    num_batch = 0    
    with torch.no_grad():
        for data in test_loader:  
            test_x = data[:, :-1].to(torch.float32)
            test_y = data[:, -1].to(torch.long)
            output = model(test_x)            
            test_loss += loss_func(output, test_y).item()
            softmax_idx = F.softmax(output, dim=1)
            max_prediction = torch.max(softmax_idx,1)
            prediction = max_prediction[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = test_y.numpy()
            if num_batch==0:
                pred_list = pred_y
            else:
                pred_list = np.concatenate((pred_list, pred_y), axis=0)
            if num_batch==0:
                target_list = target_y
            else:
                target_list = np.concatenate((target_list, target_y), axis=0)
            num_batch+=1      

        test_loss = test_loss/num_batch        
        acc = sum(pred_list == target_list)/(test_size)
        auc = roc_auc_score(target_list, pred_list)
        print('*************************')
        print('test_loss', test_loss)
        print('Test acc: ', acc)
        print('Test auc: ', auc)        
        print('*************************')
        writer.add_scalar('test_loss',test_loss, epoch)
        writer.add_scalar('auc', auc, epoch)
        writer.add_scalar('acc', acc, epoch)

model = Classifier(len(data[0])-1, 2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):    
    train(model, train_stable_loader ,train_unstable_loader , optimizer, epoch)
    test(model, test_loader, epoch)

writer.close()