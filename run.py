import os
import torch.nn as nn
import torch.optim as optim
import torch
from parameter import parse_args,setuplogging
import numpy as np
import struct
from sklearn.metrics import f1_score
import copy
import wandb
from pathlib import Path
import logging
import random
from torch.utils.data import TensorDataset,DataLoader
from torchvision import transforms 
import torch.nn.functional as F
from resnet9 import ResNet9 as base_model
from load_data import load_data_cifar as load_data

args = parse_args()
Path(args.model_dir).mkdir(parents=True, exist_ok=True)
Path(args.log_dir).mkdir(parents=True, exist_ok=True)
setuplogging(args,0)
logging.info(args)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)
#-------------------------load data---------------------#
train_images, train_labels = load_data('train')
test_images, test_labels = load_data('test')
train_images = torch.FloatTensor(train_images).cuda()
test_images = torch.FloatTensor(test_images).cuda()
#-------------------------data partition---------------------#
range_length = len(train_labels)//args.num_users
CLASS_NUM = len(set(train_labels))
alpha = args.alpha

category_dict = {}
category_used_data = {}
train_users_dict = {}

LABELED_DATA_RATIO = args.label_ratio
LABELED_DATA_NUM = int(LABELED_DATA_RATIO*range_length)
UNLABELED_DATA_NUM = int(args.unlabeled_data_ratio*(range_length-LABELED_DATA_NUM))
print(LABELED_DATA_NUM,UNLABELED_DATA_NUM)


for cid in range(CLASS_NUM):
    category_used_data[cid] = 0
    flag = np.where(train_labels == cid)[0]
    perb = np.random.permutation(len(flag))
    flag = flag[perb]
    category_dict[cid] = flag
    
user_num = int(np.ceil(len(train_labels)//range_length))
for uid in range(user_num-1):
    train_users_dict[uid] = []
    
    p = np.random.dirichlet([alpha]*CLASS_NUM, size=1)*range_length
    p = np.array(np.round(p),dtype='int32')[0]
    ix = p.argmax()
    p[ix] = p[ix] - (p.sum()-range_length)
    assert p.sum() == range_length and (p>=0).mean() == 1.0
    
    data = []
    for cid in range(CLASS_NUM):
        s = category_used_data[cid]
        ed = s + p[cid]
        category_used_data[cid] += p[cid]
        
        data.append(category_dict[cid][s:ed])
    data = np.concatenate(data).tolist()
    
    if len(data)<range_length:
        for cid in range(len(category_used_data)):
            left = range_length-len(data)
            if category_used_data[cid] < category_dict[cid].shape[0]:
                s = category_used_data[cid]
                ed = s + left
                category_used_data[cid] += left
                data += category_dict[cid][s:ed].tolist()
            left = range_length-len(data) 
            if left ==0:
                break
    data = np.array(data)
    train_users_dict[uid] = data
    
    
data = []
for cid in range(CLASS_NUM):
    s = category_used_data[cid]        
    data.append(category_dict[cid][s:])
data = np.concatenate(data)
train_users_dict[user_num-1] = data

train_users = train_users_dict

for uid in train_users:
    index = np.random.permutation(len(train_users[uid]))
    train_users[uid] = train_users[uid][index]

train_labels = torch.tensor(train_labels).cuda()
test_labels = torch.tensor(test_labels).cuda()

#-------------------------train model---------------------#
def train(epoch,global_model,global_model_weights,epoch1=1,args=None):    
    
    all_updates = []
    select_client = random.sample(train_users.keys(),int(args.sample_ratio*args.num_users))

    for uid in select_client:
        sample_indexs = train_users_dict[uid]

        global_model.load_state_dict(global_model_weights)
        global_model.train()

        local_model = user_model[uid]
        local_model.cuda()
        local_model.train()

        local_ema = copy.deepcopy(local_model.state_dict())

        for key,parameter in local_ema.items():
            if 'num_batches_tracked' in key:
                local_ema[key] = parameter+0
            else:
                local_ema[key] = args.ema_weight*local_ema[key] + (1-args.ema_weight)*global_model_weights[key]

        local_model.load_state_dict(local_ema)
        user_model[uid] = local_model

        global_optimizer =  optim.SGD([{'params':global_model.parameters(), 'lr':0.01}])
        local_optimizer = optim.SGD([{'params':local_model.parameters(), 'lr':0.01}])

        labeled_sample_indexs = sample_indexs[:LABELED_DATA_NUM]
        label_data = train_images[labeled_sample_indexs]
        label_data_y = train_labels[labeled_sample_indexs]
        label_shape=label_data.shape[0]
        while args.per_device_train_batch_size>label_data.shape[0]:
            label_data = torch.cat([label_data,label_data],0)
            label_data_y = torch.cat([label_data_y,label_data_y],0)

        label_dataset = TensorDataset(label_data,label_data_y)
        label_dataloader = DataLoader(dataset = label_dataset,batch_size = args.per_device_train_batch_size,shuffle=True,drop_last = True)

        unlabeled_sample_indexs = sample_indexs[LABELED_DATA_NUM:LABELED_DATA_NUM+UNLABELED_DATA_NUM]
        unlabel_data = train_images[unlabeled_sample_indexs]
        if UNLABELED_DATA_NUM==0:
            epoch1 = (range_length-LABELED_DATA_NUM)//(args.per_device_train_batch_size)//(label_data.shape[0]//args.per_device_train_batch_size)
            for j in range(epoch1):
                for l_xy in label_dataloader:
                    l_x,l_y = l_xy
                    weak_aug_x = weak_aug(l_x)
                    strong_aug_x = strong_aug((l_x*255).type(torch.uint8))/255
                    local_pred,local_loss = local_model(weak_aug_x,l_y)
                    global_pred,global_loss = global_model(strong_aug_x,l_y)

                    local_loss[torch.isinf(local_loss)]=0
                    local_loss[torch.isnan(local_loss)]=0
                    global_loss[torch.isinf(global_loss)]=0
                    global_loss[torch.isnan(global_loss)]=0

                    loss_label = torch.mean(local_loss) + torch.mean(global_loss)
                    loss = loss_label

                    global_optimizer.zero_grad()
                    local_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(global_model.parameters(),0.1)
                    nn.utils.clip_grad_value_(local_model.parameters(),0.1)
                    global_optimizer.step()
                    local_optimizer.step()
                    
            updates = copy.deepcopy(global_model.state_dict())
            all_updates.append(updates)
            local_model.cpu()
            continue
        unlabel_dataset = TensorDataset(unlabel_data)
        unlabel_dataloader = DataLoader(dataset = unlabel_dataset,batch_size = args.per_device_train_batch_size,shuffle=True,drop_last = True)
        epoch1 = unlabel_data.shape[0]//(args.per_device_train_batch_size)//max((label_shape//args.per_device_train_batch_size),1)
        for j in range(epoch1):
            for l_xy,u_x in zip(label_dataloader,unlabel_dataloader):
                u_x = u_x[0]
                l_x,l_y = l_xy
                weak_aug_x = weak_aug(l_x)
                strong_aug_x = strong_aug((l_x*255).type(torch.uint8))/255
                local_pred,local_loss = local_model(weak_aug_x,l_y)
                global_pred,global_loss = global_model(strong_aug_x,l_y)

                local_loss[torch.isinf(local_loss)]=0
                local_loss[torch.isnan(local_loss)]=0
                global_loss[torch.isinf(global_loss)]=0
                global_loss[torch.isnan(global_loss)]=0

                loss_label = torch.mean(local_loss) + torch.mean(global_loss)

                weak_aug_x = weak_aug(u_x)
                strong_aug_x = strong_aug((u_x*255).type(torch.uint8))/255

                global_pred = global_model(weak_aug_x)[0]
                local_pred = local_model(weak_aug_x)[0]

                local_pred_prob = F.softmax(local_pred,-1)
                global_pred_prob = F.softmax(global_pred,-1)

                global_pred_strong = global_model(strong_aug_x)[0]
                local_pred_strong = local_model(strong_aug_x)[0]

                local_pred_prob_strong = F.softmax(local_pred_strong,-1)
                global_pred_prob_strong = F.softmax(global_pred_strong,-1)

                global_entropy = -torch.sum(global_pred_prob*torch.log(global_pred_prob+1e-8),-1)
                local_entropy = -torch.sum(local_pred_prob*torch.log(local_pred_prob+1e-8),-1)
                local_teacher = local_entropy<args.entropy_threshold
                local_teacher = local_teacher.float().clone().detach()
                global_teacher = global_entropy<args.entropy_threshold
                global_teacher = global_teacher.float().clone().detach()

                global_teacher_loss = global_teacher*torch.sum(global_pred_prob.detach()*torch.log(global_pred_prob.detach()/(local_pred_prob_strong+1e-8)+1e-8),-1) 
                local_teacher_loss = local_teacher*torch.sum(local_pred_prob.detach()*torch.log(local_pred_prob.detach()/(global_pred_prob_strong+1e-8)+1e-8),-1) 

                global_teacher_loss[torch.isinf(global_teacher_loss)] = 0
                global_teacher_loss[torch.isnan(global_teacher_loss)] = 0
                local_teacher_loss[torch.isinf(local_teacher_loss)] = 0
                local_teacher_loss[torch.isnan(local_teacher_loss)] = 0

                loss_unlabel = torch.mean(global_teacher_loss) + torch.mean(local_teacher_loss)
                loss = loss_label + min(1,epoch/200)*loss_unlabel

                global_optimizer.zero_grad()
                local_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(global_model.parameters(),0.1)
                nn.utils.clip_grad_value_(local_model.parameters(),0.1)
                global_optimizer.step()
                local_optimizer.step()

        updates = copy.deepcopy(global_model.state_dict())
        all_updates.append(updates)
        local_model.cpu()

        
    sup_aggregate = copy.deepcopy(global_model.state_dict())
    for key,parameter in sup_aggregate.items():
        if 'num_batches_tracked' in key:
            sup_aggregate[key] = parameter+0
        else:
            sup_aggregate[key] = torch.zeros_like(parameter)

    for uid,update in zip(select_client,all_updates):
        for key,parameter in update.items():
            if 'num_batches_tracked' in key:
                sup_aggregate[key] = parameter+0
            else:
                sup_aggregate[key] += parameter/len(select_client)

    

    return sup_aggregate

weak_aug = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')
                                ])
strong_aug=transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
                                  transforms.RandAugment()
                                ])

global_model = base_model(3,10)
user_model = [copy.deepcopy(global_model) for i in range(args.num_users)]
global_model.cuda()
global_model_weights = copy.deepcopy(global_model.state_dict())
best_acc = 0
for i in range(args.rounds):

    global_model_weights = train(i,global_model,global_model_weights,1,args)
    
    global_model.eval()
    global_model.load_state_dict(global_model_weights)
    with torch.no_grad():
        pred = global_model(test_images)[0].detach().argmax(axis=-1)
        sup_correct = (pred==test_labels)
        sup_correct = sup_correct.float()
    
    logging.info(f'epoch {i} test acc : {sup_correct.mean().item()}')


