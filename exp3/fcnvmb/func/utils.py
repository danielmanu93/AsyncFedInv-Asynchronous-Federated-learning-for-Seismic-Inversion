# -*- coding: utf-8 -*-
"""
Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""
import torch
import numpy as np
import torch.nn as nn
from math import log10
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

def turn(GT):
    dim = GT.shape
    for j in range(0,dim[1]):
        for i in range(0,dim[0]//2):
            temp    = GT[i,j]
            GT[i,j] = GT[dim[0]-1-i,j]
            GT[dim[0]-1-i,j] = temp
    return GT 


def PSNR(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target     = Variable(torch.from_numpy(target))
    zero       = torch.zeros_like(target)   
    criterion  = nn.MSELoss(size_average=True)    
    MSE        = criterion (prediction, target)
    total      = criterion (target, zero)
    psnr       = 10. * log10(total.item() / MSE.item())
    return psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1    = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2    = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L  = 255
    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def SaveTrainResults(loss,SavePath,font2,font3):
    fig,ax  = plt.subplots()
    plt.plot(loss[1:], linewidth=2)
    plt.plot(loss, linewidth=2)
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('MSE Loss', font2)
    ax.set_title('Training', font3)
    ax.set_xlim([1,500])
    #ax.set_xticklabels(('0','1000','2000','3000','4000','5000'))
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed',linewidth=0.5)
    
     
    plt.savefig(SavePath+'TrainLoss', transparent = True)
    data = {}
    data['loss'] = loss
    #scipy.io.savemat(SavePath+'TrainLoss',data)
    np.save(SavePath+'TrainLoss.npy',data)
    plt.show()
    plt.close()

def SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,SavePath):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM    
    data['GT']      = GT
    data['Prediction'] = Prediction
    #scipy.io.savemat(SavePath+'TestResults',data) 
    np.save(SavePath+'TestResults.npy',data)
    
    
def PlotComparison(pd,gt,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath):
    PD = pd.reshape(label_dsp_dim[0],label_dsp_dim[1])
    GT = gt.reshape(label_dsp_dim[0],label_dsp_dim[1])
    fig1,ax1 = plt.subplots(figsize=(6, 4))    
    im1     = ax1.imshow(GT,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)
    divider = make_axes_locatable(ax1)
    cax1    = divider.append_axes("right",size="5%",pad=0.05)
    plt.colorbar(im1,ax=ax1,cax=cax1).set_label('Velocity (m/s)')
    plt.tick_params(labelsize=12)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(14)
    ax1.set_xlabel('Position (km)',font2)
    ax1.set_ylabel('Depth (km)',font2)
    ax1.set_title('Ground truth',font3)
    ax1.invert_yaxis()
    plt.subplots_adjust(bottom=0.15,top=0.92,left=0.08,right=0.98)
    plt.savefig(SavePath+'GT',transparent=True)
    
    fig2,ax2=plt.subplots(figsize=(6, 4))
    im2=ax2.imshow(PD,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)

    plt.tick_params(labelsize=12)  
    for label in  ax2.get_xticklabels()+ax2.get_yticklabels():
        label.set_fontsize(14)   
    ax2.set_xlabel('Position (km)',font2)
    ax2.set_ylabel('Depth (km)',font2)
    ax2.set_title('Prediction',font3)
    ax2.invert_yaxis()
    plt.subplots_adjust(bottom=0.15,top=0.92,left=0.08,right=0.98)
    plt.savefig(SavePath+'PD',transparent=True)
    plt.show()
    plt.close()
   
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# Compute local and global model norm difference
def model_norm_diff(model_local_params, model_global_params):
    weight = 1  # 0.01 
    for key in model_local_params:
        if key in model_global_params:
            model_local_params[key] -= model_global_params[key]
    
    # compute the L2 norm of the weights
    l2_norm = 0
    for key, value in model_local_params.items():
        if "weight" in key:
            l2_norm += torch.norm(value, p=2, dtype=torch.float)
    l2_norm = weight * l2_norm.item()
    return l2_norm


def data_iid(dataset, NumUsers):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    user_samples = []
    num_items = int(len(dataset)/NumUsers)
    user_samples.extend([num_items] * NumUsers)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(NumUsers):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                            replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return user_samples

def data_noniid(dataset, NumUsers):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    beta = 0.1 # Can be 0.1, 1 or 10
    num_shards, num_imgs = 3, 280
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(NumUsers)}
    idxs = np.arange(num_shards*num_imgs)
    idxs = np.reshape(idxs, (idxs.shape[0], 1))

    labels = []
    for _, label in dataset:
        labels.append(label.numpy())
    # print("labels:", labels)
    labels = np.array(labels)
    labels = labels.reshape(-1, labels.shape[-1])
    # print(labels)
    # print(idxs)

    # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    idxs_labels = np.concatenate((idxs, labels), axis=1)
    idxs_labels = idxs_labels.astype('int32')
    # print(idxs_labels)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[:, 0]
    # print(idxs)

    dirichlet_params = np.repeat(beta, NumUsers)
    proportions = np.random.dirichlet(dirichlet_params, size=NumUsers)
    proportions = proportions / proportions.sum()
    proportions = np.cumsum(proportions)
    props = [round(i, 2) for i in proportions]
    new_props = np.unique(props)

    # divide and assign 2 shards/client
    # for i in range(NumUsers):
    #     rand_set = set(np.random.choice(idx_shard, 1, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

    user_samples = []
    for j in range(NumUsers):
        rand_set = set(np.random.choice(new_props[1:], 1, replace=False))
        # print(rand_set)
        ind = np.where(new_props == list(rand_set))
        # print(ind)
        # next_prop = new_props[ind[0][0] + 1]
        prev = new_props[ind[0][0]-1]
        for rand in rand_set:
            # prev_idx = int(prev*num_mols)
            # curr_idx = int(rand*num_mols)
            # num_samples = abs(curr_idx - prev_idx)
            # print(num_samples)
            # print(idxs[int(prev*num_mols)])
            # print(idxs[int(rand*num_mols)])
            num_samples = abs(idxs[int(prev*num_imgs)] - idxs[int(rand*num_imgs)])
            user_samples.append(num_samples)
            # print(user_samples)
        
            dict_users[j] = np.concatenate((dict_users[j], idxs[int(prev*num_imgs):int(rand*num_imgs)]), axis=0)

    return user_samples
