import numpy as np
import os

root=os.getenv("AMLT_DATA_DIR", "../")

def load_data_mnist(mode):
    if mode == 'train':
        file_path = '/workspaceblobstore/v-chaozhang/semiKD/data/train-images-idx3-ubyte'
        label_path = '/workspaceblobstore/v-chaozhang/semiKD/data/train-labels-idx1-ubyte'
    else:
        file_path = '/workspaceblobstore/v-chaozhang/semiKD/data/t10k-images-idx3-ubyte'
        label_path = '/workspaceblobstore/v-chaozhang/semiKD/data/t10k-labels-idx1-ubyte'
    
    binfile = open(file_path, 'rb') 
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    
    images = images.reshape((len(images),1,28,28))
    
    binfile = open(label_path, 'rb')
    buffers = binfile.read()
    magic,num = struct.unpack_from('>II', buffers, 0) 
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    
    images = images/255
    
    return images,labels


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