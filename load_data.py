import numpy as np
import os
import torch
import scipy.io as sio
import torchvision
from datasets import load_dataset

root="../"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_cifar(mode):
    if mode=='train':
        x=[]
        y=[]
        for i in range(1,6):
            batch_path=os.path.join(root,'data/cifar-10-batches-py/data_batch_%d'%(i))
            batch_dict=unpickle(batch_path)
            train_batch=batch_dict[b'data'].astype('float')
            train_labels=np.array(batch_dict[b'labels'])
            x.append(train_batch)
            y.append(train_labels)

        images=np.concatenate(x)
        labels=np.concatenate(y)

    
    if mode == 'test':
        testpath=os.path.join(root,'data/cifar-10-batches-py/test_batch')
        test_dict=unpickle(testpath)
        images=test_dict[b'data'].astype('float')
        labels=np.array(test_dict[b'labels'])

    images = images.reshape((len(images),3,32,32))
    images = images/255
    
    return images,labels

def load_data_cifar100(mode):
    if mode=='train':
        dataset = torchvision.datasets.CIFAR100(root='/data/semifl/data', train=True, download=True)
    
    if mode == 'test':
        dataset = torchvision.datasets.CIFAR100(root='/data/semifl/data', train=False, download=True)

    
    images = np.zeros((len(dataset), 3, 32, 32), dtype=np.float32)
    labels = np.zeros((len(dataset),), dtype=np.int64)
    for i, (inputs, label) in enumerate(dataset):
        images[i] = np.array(inputs).transpose(2,0,1)
        labels[i] = label
        
    images = images/255
    
    return images,labels

def load_data_TinyImageNet(mode):
    if mode=='train':
        dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    
    if mode == 'test':
        dataset = load_dataset('Maysee/tiny-imagenet', split='valid')
    
    images = np.zeros((len(dataset), 3, 32, 32), dtype=np.float32)
    labels = np.zeros((len(dataset),), dtype=np.int64)
    for i, data in enumerate(dataset):
        inputs = data['image'].resize((32,32))
        label = data['label']
        if len(inputs.split()) != 3:
            inputs = inputs.convert('RGB')

        images[i] = np.array(inputs).transpose(2,0,1)
        labels[i] = label
    images = images/255
    
    return images,labels

def load_data_svhn(mode):
    if mode =='train':
        train = sio.loadmat(os.path.join(root,'data/svhn/train_32x32.mat'))
        images = train['X']
        labels = train['y']
        images = np.swapaxes(images, 0, 3)
        images = np.swapaxes(images, 1, 2)
        images = np.swapaxes(images, 2, 3)
        labels = labels.reshape(73257, )
        labels = np.array(labels)
        labels[labels==10] = 0 

    if mode == "test":
        test = sio.loadmat(os.path.join(root,'data/svhn/test_32x32.mat'))
        images = test['X']
        labels = test['y']
        images = np.swapaxes(images, 0, 3)
        images = np.swapaxes(images, 1, 2)
        images = np.swapaxes(images, 2, 3)
        labels = labels.reshape(26032, )
        labels = np.array(labels)
        labels[labels==10] = 0 
    

    images = images/255
    
    return images,labels
