# -*- coding: utf-8 -*-
"""
Fully Convolutional neural network (U-Net) for velocity model building from prestack unmigrated seismic data directly

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

def model_averaging(alpha, local_weights, global_weights):
    for key in local_weights:
        # print("local weights: ", local_weights)
        local_weights[key] = local_weights[key] * torch.tensor(alpha, dtype=local_weights[key].dtype)
        # print("updated local weights: ", local_weights)
    for key in global_weights:
        # print("global weights: ", global_weights)
        global_weights[key] = global_weights[key] * torch.tensor((1 - alpha), dtype=global_weights[key].dtype)
        # print("updated global weights: ", global_weights)

    for key in global_weights:
        if key in local_weights:
            global_weights[key] += local_weights[key]
    return global_weights

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device         = torch.device("cuda" if cuda_available else "cpu")

net = UnetModel(n_classes=Nclasses,in_channels=Inchannels,is_deconv=True,is_batchnorm=True) 
if torch.cuda.is_available():
    net.cuda()

def get_alpha(a, b, alpha, staleness, staleness_method):
    if staleness_method == "constant":
        return torch.mul(alpha, 1)
    elif staleness_method == "exponential" and a is not None:
        return torch.mul(alpha, math.exp(-a * (staleness)))
    elif staleness_method == "hinge" and a is not None and b is not None:
        if staleness <= b:
            return torch.mul(alpha, 1)
        else:
            return torch.mul(alpha, math.exp(-math.log(abs(a * (staleness - b)) + 1)))

def compute(num_users):
    np.random.seed(seed=0)
    cycleBit = np.random.uniform(1, 5, num_users) * 10 ** 8
    np.random.seed(seed=0)
    compFreq = np.random.uniform(0.1, 1, num_users) * 10 ** 8
    return cycleBit, compFreq

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
        data_iid(train_loader, NumUsers)

    else:
        data_noniid(train_loader, NumUsers)

    return train_loader

train_data = get_loader()

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

def train(net, global_round):

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
            loss1 = np.append(loss1, total_epoch_loss)
            time_elapsed = time.time() - since
            print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
        # Save net parameters every 10 epochs
        if (epoch+1) % SaveEpoch == 0:
            torch.save(net.state_dict(),models_dir+modelname+'_epoch'+str(epoch+1)+'.pkl')
            print ('Trained model saved: %d percent completed'% int((epoch+1)*100/LocalEpochs))
    
    return net.state_dict(), loss1

global_model = net

# copy global weights
global_weights = net.state_dict()

train_loss = []
last_local_loss = []

train_users = max(int(frac * NumUsers), 1)

idxs_users = [user for user in range(train_users)]

local_models = []

for idx in idxs_users:
    local_model = net
    local_models.append(local_model)

# compute the cycle bit and freq of each user
cycleBit, compFreq = compute(NumUsers)
user_samples = data_noniid(train_data, NumUsers)
print(user_samples)

# compute latency based on the model training -> (C * d)/f 
compLatency = [(C * d)/f for C, d, f in zip(cycleBit, user_samples, compFreq)]
print("Comp latency : ", compLatency)
min_latency = min(compLatency)
print("Min latency : ", min_latency)

skipped_latency = []
model_skips = 0

for i in tqdm(range(GlobalEpochs)):

    local_losses = []

    print(f'\n | Global Training round : {i+1} |\n')

    last_local_loss.clear()

    for idx in idxs_users:
        
        local_weights, model_loss = train(local_models[idx], global_round=i)
        
        # get train loss
        # local_losses.append(copy.deepcopy(model_loss)) 
        print("client_loss: ", model_loss)

        last_local_loss.append(model_loss[-1])
        print("client last loss: ", last_local_loss)        

        # compute staleness based on the minimum latency 
        staleness = compLatency[idx] - min_latency
        print("Staleness : ", staleness)

        # compute alpha_t
        alpha_t = get_alpha(a, b, alpha, staleness, staleness_method)

        # Compute model averaging before updating global model
        new_global_weights = model_averaging(alpha_t, local_weights, global_weights)

        # update global weights
        global_model.load_state_dict(new_global_weights)

        global_weights = global_model.state_dict()

        norm_diff = model_norm_diff(local_weights, global_weights)

        print("Model norm diff", norm_diff)    

        if norm_diff <= epsilon:

            # do not train the local model in the next round
            print(f"Skipping training the model for user {idx} in round {i+1}")
            model_skips += 1

            compLatency[idx] = 0
            skipped_latency.append(compLatency[idx])
            min_latency = min(skipped_latency)
            print("min_skipped_client_latency: ", min_latency)

            local_weights = local_weights

            local_models[idx].load_state_dict(local_weights)

        elif norm_diff > epsilon:
            print("cycleBit : ", cycleBit)
            print("user_samples : ", user_samples)
            print("compFreq : ", compFreq)

            compLatency[idx] = [(C * d)/f for C, d, f in zip([cycleBit[idx]], [user_samples[idx]], [compFreq[idx]])]
            compLatency[idx] = compLatency[idx][0]

            local_weights = global_weights
            local_models[idx].load_state_dict(local_weights)

    local_loss = np.array(last_local_loss).ravel()

    # average last local losses of all clients - global round loss
    loss_avg = sum(local_loss) / len(local_loss)

    # append global round loss into train_loss 
    train_loss.append(loss_avg)

    train_loss_array = np.array(train_loss)

    np.savetxt("/vast/home/dmanu/Desktop/exp3/fcnvmb/model loss for {} function with epsilon={}.txt".format(staleness_method, epsilon), train_loss_array)

print("Total model skips: ", model_skips)

plt.figure(figsize=(10, 5))
plt.plot(range(GlobalEpochs), train_loss)
plt.xticks(np.arange(0, GlobalEpochs, 10), fontsize=14)
plt.xlabel("Global rounds", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Model Loss for {} function, epsilon={}, alpha={}".format(staleness_method, epsilon, alpha))
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
