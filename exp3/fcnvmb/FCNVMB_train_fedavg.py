# -*- coding: utf-8 -*-
"""
Fully Convolutional neural network (U-Net) for velocity model building from prestack unmigrated seismic data

@author: Daniel Manu (dmanu@unm.edu)
"""

################################################
########        IMPORT LIBARIES         ########
################################################

from ParamConfig import *
from PathConfig import *
from LibConfig import *
import torch
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
from func import utils
################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device         = torch.device("cuda" if cuda_available else "cpu")

net = UnetModel(n_classes=Nclasses,in_channels=Inchannels,is_deconv=True,is_batchnorm=True) 
if torch.cuda.is_available():
    net.cuda()

def print_network(model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  #get no. of params iteratively
        # print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

print_network(net, "Net_model")

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(),lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading the pre-trained model *****************')
    print('')
    premodel_file = models_dir + premodelname + '.pkl'
    ##Load generator parameters
    net  = net.load_state_dict(torch.load(premodel_file))
    net  = net.to(device)
    print('Finish downloading:',str(premodel_file))
    
################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading Training DataSet *****************')
def get_loader():
    global data_dsp_dim, label_dsp_dim
    train_set,label_set,data_dsp_dim,label_dsp_dim  = DataLoad_Train(train_size=TrainSize,train_data_dir=train_data_dir, \
                                                                    data_dim=DataDim,in_channels=Inchannels, \
                                                                    model_dim=ModelDim,data_dsp_blk=data_dsp_blk, \
                                                                    label_dsp_blk=label_dsp_blk,start=1, \
                                                                    datafilename=datafilename,dataname=dataname, \
                                                                    truthfilename=truthfilename,truthname=truthname)

    # print(label_set)
    # Change data type (numpy --> tensor)
    dataset        = data_utils.TensorDataset(torch.from_numpy(train_set),torch.from_numpy(label_set))
    train_loader   = data_utils.DataLoader(dataset,batch_size=BatchSize,shuffle=True)

    if data_distrib == 1:
        user_groups = data_iid(train_loader, NumUsers)

    else:
        user_groups = data_noniid(train_loader, NumUsers)

    return train_loader, user_groups

train_data, user_groups = get_loader()
print(user_groups)
################################################
########            TRAINING            ########
################################################

print() 
print('*******************************************') 
print('*******************************************') 
print('           START TRAINING                  ') 
print('*******************************************') 
print('*******************************************') 
print() 

print ('Original data dimention:%s'      %  str(DataDim))
print ('Downsampled data dimention:%s '  %  str(data_dsp_dim))
print ('Original label dimention:%s'     %  str(ModelDim))
print ('Downsampled label dimention:%s'  %  str(label_dsp_dim))
print ('Training size:%d'                %  int(TrainSize))
print ('Traning batch size:%d'           %  int(BatchSize))
print ('Number of epochs:%d'             %  int(LocalEpochs))
print ('Learning rate:%.5f'              %  float(LearnRate))

start  = time.time()

def train(net, global_round, idxs):

    global loss1

    # Initialization
    loss1  = 0.0
    step   = np.int(TrainSize/BatchSize)

    for epoch in range(LocalEpochs): 
        epoch_loss = 0.0
        since      = time.time()

        for i, (images,labels) in enumerate(train_data):        
            iteration  = epoch*step+i+1
            # Set Net with train condition
            net.train()
            
            # Reshape data size
            images = images.view(BatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            labels = labels.view(BatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the gradient buffer
            optimizer.zero_grad()     
            
            # Forward prediction
            outputs = net(images, label_dsp_dim)

            # Calculate the MSE
            loss = F.mse_loss(outputs,labels,reduction='sum')/(label_dsp_dim[0]*label_dsp_dim[1]*BatchSize)
            # loss = loss / 10 ** 5

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')
                
            epoch_loss += loss.item()

            # Loss backward propagation    
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Print loss
            if iteration % DisplayStep == 0:
                print('Global Round: {} | Epoch: {}/{} | Iteration: {}/{} | Training Loss:{:.6f}'.format(global_round, epoch+1, \
                                                                                                        LocalEpochs,iteration, \
                                                                                                        step*LocalEpochs,loss.item()))        

        # Print loss and consuming time every epoch
        if (epoch+1) % 1 == 0:
            #print ('Epoch [%d/%d], Loss: %.10f' % (epoch+1,Epochs,loss.item()))          
            #loss1 = np.append(loss1,loss.item())
            total_epoch_loss = epoch_loss / i
            print('Epoch: {:d} finished ! Loss: {:.5f}'.format(epoch+1,total_epoch_loss))
            loss1 = np.append(loss1,total_epoch_loss)
            time_elapsed = time.time() - since
            print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
        # Save net parameters every 10 epochs
        if (epoch+1) % SaveEpoch == 0:
            torch.save(net.state_dict(),models_dir+modelname+'_epoch'+str(epoch+1)+'.pkl')
            print ('Trained model saved: %d percent completed'% int((epoch+1)*100/LocalEpochs))
    
    return net.state_dict(), loss1

global_model = net

# copy global weights
global_model_weights = net.state_dict()

train_loss = []
last_net_loss = []

for i in tqdm(range(GlobalEpochs)):

    local_net_weights, local_net_losses = [], []

    print(f'\n | Global Training round : {i+1} |\n')

    train_users = max(int(frac * NumUsers), 1)

    idxs_users = np.random.choice(range(NumUsers), train_users, replace=False)

    last_net_loss.clear()
    
    for idx in idxs_users:
        
        net_weights, net_loss = train(net, global_round=i, idxs=user_groups[idx])

        local_net_weights.append(copy.deepcopy(net_weights))
        # local_net_losses.append(copy.deepcopy(net_loss)) 

        last_net_loss.append(net_loss[-1])
        print(last_net_loss)

    net_local_loss = np.array(last_net_loss).ravel()

    # average local weights
    global_net_weights = utils.average_weights(local_net_weights)
    
    # update global weights
    global_model.load_state_dict(global_model_weights)

    # average last local losses of all clients - global loss
    net_loss_avg = sum(net_local_loss) / len(net_local_loss)

    # append global loss into train_loss 
    train_loss.append(net_loss_avg)

    net_train_loss_array = np.array(train_loss)

    np.savetxt("/vast/home/dmanu/Desktop/exp5/fcnvmb/net loss for FedAvg_[32, 64, 128, 256, 512].txt", net_train_loss_array)

plt.figure(figsize=(10, 5))
plt.plot(range(GlobalEpochs), train_loss)
# plt.xticks(np.arange(0, GlobalEpochs, 5), fontsize=14)
plt.xlabel("Global rounds", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend()
plt.show()

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s' .format(time_elapsed //60 , time_elapsed % 60))

# Save the loss
font2  = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 17,
    }
font3 = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
    }
SaveTrainResults(loss=loss1,SavePath=results_dir,font2=font2,font3=font3)
