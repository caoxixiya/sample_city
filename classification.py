import torch
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F

data = np.load('city1_data.npy', allow_pickle=True)

train_size = int(len(data)*0.7)
BATCH_SIZE = 2560
EPOCHS = 100

global test_size
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data_batch in enumerate(train_loader):
        train_x = data_batch[:, :-1].to(torch.float32)
        train_y = data_batch[:, -1].to(torch.long)
        optimizer.zero_grad()               
        output = model(train_x)                
        loss = loss_func(output, train_y)   
        loss.backward()                     
        optimizer.step()                    
        if batch_idx%30 == 0:            
            # print('train loss:', loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    test_loss = 0                           
    correct = 0                             
    with torch.no_grad():           
        for data in test_loader:               
            test_x = data[:, :-1].to(torch.float32)
            test_y = data[:, -1].to(torch.long)
            output = model(test_x)            
            test_loss += loss_func(output, test_y).item()
            prediction = torch.max(F.softmax(output, dim=1),1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = test_y.numpy()
            correct += sum(pred_y == target_y)          

        accuracy = correct/test_size
        print('*************************')
        print('Test accuracy: ', accuracy)
        print('*************************')


model = Classifier(len(data[0])-1, 2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()


for epoch in range(EPOCHS):    
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
