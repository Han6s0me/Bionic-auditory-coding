from scipy import signal
import numpy as np

import torch
import torchaudio
import matplotlib.pyplot as plt

from spikingjelly.activation_based import encoding
from spikingjelly import visualizing
import os
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import torch.nn.functional as F

from librosa import cqt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import librosa
from scipy.io import loadmat

# Dataset name
train_dir = '../snndatabase/TID_DATA_8k_all/'
# train_dir = '../snndatabase/RWCP_train_8k_all/'

# Parameter of training
T = 10
batch_size = 64
learning_rate = 1e-2
tau = 4.0
train_epoch = 200
savepath='./snnmodel/Model_TID_R_MSE/BAN_encoding/T10/' # Change the savepath for model
Max_num=83

# Loading data
mat_data = loadmat('./After_encoding_data/TID/BAN_encoding/train_data_10T.mat')
mat_data_label = loadmat('./After_encoding_data/TID/BAN_encoding/label_data_10T.mat')
variable_name = 'train_data'
variable_name_label = 'train_labels'
train_data = mat_data[variable_name]
train_labels = mat_data_label[variable_name_label]
train_labels=np.squeeze(train_labels)

# Data split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Repeat the training 10 times
for id in range(10):
    random_state=(id+1)*10
    (X_train, X_test, Y_train, Y_test) = train_test_split(train_data, train_labels, test_size=0.1, random_state=random_state)
 
    def data_package(train_X, train_Y, test_X, test_Y, batch_size):
    
        x_train_data = []

        x_train_data = torch.from_numpy(train_X).type(torch.FloatTensor)
        y_train_data = torch.from_numpy(train_Y).type(torch.FloatTensor)
        x_test_data = []

        x_test_data = torch.from_numpy(test_X).type(torch.FloatTensor)
        y_test_data = torch.from_numpy(test_Y).type(torch.FloatTensor)

        train_dataset = TensorDataset(x_train_data, y_train_data)
        test_dataset = TensorDataset(x_test_data, y_test_data)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


    # SNN model
    class SNNforauditory(nn.Module):
        def __init__(self, tau):
            super().__init__()

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(83 * Max_num, 1024, bias=False),
                neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                layer.Dropout(0.5),

            )
            self.fc1 = nn.Sequential(
                nn.Linear(1024, 11, bias=False),  # For TID dataset
                # nn.Linear(1024, 10, bias=False), # For RWCP dataset
                neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

            )

        def forward(self, x: torch.Tensor):
            x = self.fc(x)
            x = self.fc1(x)
            return x

    # Encapsulation data
    train_loader, test_loader = data_package(X_train, Y_train, X_test, Y_test, batch_size)
    # The format of network training data is [N, t, F, T]. t is the speech time, F is the frequency, T is the neuron simulation time, and N is the sample

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print('Total GPU numbers:' + str(torch.cuda.device_count()), '\n', 'Being uesd GPU:' + str(device))
    torch.cuda.manual_seed(0)
    net = SNNforauditory(tau=tau).to(device)
    log_dir = ''
    # Using AdamW optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    train_times = 0
    max_test_accuracy = -1

    train_accuracy_all = []
    train_loss_all = []

    validation_accuracy_all = []
    validation_loss_all = []

    for epoch in range(train_epoch):
        print('------------Epoch:%s------------' % (epoch + 1))
        net.train()

        train_correct_sum = 0
        train_sum = 0

        epoch_train_accuracy = []
        train_losses = []

        for batch, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            train_times += 1  # 训练次数，一个batch算一次
            data = data.to(device)

            label = label.long().to(device)

            label_one_hot = F.one_hot(label, 11).float() # For TID dataset
            # label_one_hot = F.one_hot(label, 10).float() # For RWCP dataset
    
            data_seq = data.permute(3, 0, 2, 1)
            out_spikes_counter_frequency = 0
    
            print(data.shape)
            for t in range(T):
  
                out_spikes_counter_frequency += net(data_seq[t])
            out_spikes_counter_frequency = out_spikes_counter_frequency / T

         
            # The loss function is the pulse firing frequency of the output layer neurons and the MSE of the true category
            # Such a loss function will make the pulse firing frequency of the i-th neuron in the output layer approach 1 when category i is input, while the pulse firing frequency of other neurons approaches 0.
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)

            loss.backward()
            optimizer.step()
            # After optimizing the parameters once, the state of the network needs to be reset, because the neurons of SNN have "memory"
            functional.reset_net(net)
            train_losses.append(loss.item())

            # The accuracy rate is calculated as follows. The subscript i of the neuron with the highest pulse emission frequency in the output layer is considered to be the classification result
            train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label).float().sum().item()
            train_sum += label.numel()
            train_accuracy = train_correct_sum / train_sum

            if train_times % 1 == 0:
                print('train_times:', train_times, 'train_loss:', '%.5f' % (loss.item()), 'train_accuracy:',
                      '%.5f' % train_accuracy)

            epoch_train_accuracy.append(train_accuracy)
        epoch_train_loss = np.array(train_losses).mean()
        train_loss_all.append(epoch_train_loss)
        train_accuracy_all.append(epoch_train_accuracy[-1])

        # Validation
        net.eval()
        with torch.inference_mode():
            test_sum = 0
            correct_sum = 0
            val_losses = []
            for test_data, test_label in test_loader:
                test_data = test_data.to(device)
                test_label = test_label.long().to(device)
                test_label_one_hot = F.one_hot(test_label,11).float() # For TID dataset
                # test_label_one_hot = F.one_hot(test_label,10).float()  # For RWCP dataset

                test_data_seq = test_data.permute(3, 0, 2, 1)

                test_output = 0
               
                for t in range(T):
                 
                    test_output += net(test_data_seq[t])
                test_output = test_output / T

                val_loss = F.mse_loss(test_output, test_label_one_hot)
              
                val_losses.append(val_loss.item())
                correct_sum += (test_output.max(1)[1] == test_label).float().sum().item()  # 预测正确的样本数
                test_sum += test_label.numel()  
      
                functional.reset_net(net)

            epoch_val_loss = np.array(val_losses).mean()
            validation_loss_all.append(epoch_val_loss)
            test_accuracy = correct_sum / test_sum

            print('Epoch %s' % (epoch + 1), 'test_loss:', '%.5f' % epoch_val_loss, 'test_accuracy:',
                  '%.5f' % test_accuracy)
            validation_accuracy_all.append(test_accuracy)

            if max_test_accuracy < test_accuracy:
                max_test_accuracy = test_accuracy
        
        scheduler.step(epoch_val_loss)

 
        print(
            'dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
                train_dir, batch_size, learning_rate, T, log_dir, max_test_accuracy, train_times))

    # Visualizing training results
    epochs = range(1, len(validation_accuracy_all) + 1)
    plt.figure()
    plt.plot(epochs, train_accuracy_all, 'b', label='Training accuracy')
    plt.plot(epochs, validation_accuracy_all, 'r', label='Testing accuracy')
    plt.title('Training and testing accuracy')
    plt.legend()
    plt.show()
   
    plt.figure()
    plt.plot(epochs, train_loss_all, 'b', label='Training loss')
    plt.plot(epochs, validation_loss_all, 'r', label='Testing loss')

    plt.title('Training and testing loss')
    plt.legend()
  
    plt.show()

    now = np.datetime64('now')
    timestamp = now.astype('int64')

    # Save network
    epochs=np.linspace(1, len(validation_accuracy_all) , len(validation_accuracy_all) )
    import csv

    if os.path.exists(savepath):
        print('ready exist')
    else:
        print('ok I make it')
        os.makedirs(savepath)
    filename=savepath+'Max_num:%d'%Max_num+'_'+'acc:%.4f'%train_accuracy_all[-1]+'_'+'val_acc:%.4f'%validation_accuracy_all[-1]+'_'+str(timestamp)+'_train_result.csv'
    with open(filename, 'w', newline='') as file:
        fieldnames = ['epochs', 'train_accuracy_all','train_loss_all','validation_accuracy_all','validation_loss_all']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        
        writer.writeheader()
       
        for epochs, train_accuracy_alls, train_loss_alls,validation_accuracy_alls,validation_loss_alls in zip(epochs, np.array(train_accuracy_all),np.array(train_loss_all),np.array(validation_accuracy_all),np.array(validation_loss_all)):
            writer.writerow({'epochs': epochs, 'train_accuracy_all': train_accuracy_alls,'train_loss_all':train_loss_alls,'validation_accuracy_all':validation_accuracy_alls,'validation_loss_all':validation_loss_alls})

    modelfilename=savepath+'Max_num:%d'%Max_num+'_'+'acc:%.4f'%train_accuracy_all[-1]+'_'+'val_acc:%.4f'%validation_accuracy_all[-1]+'_'+str(timestamp)+'.pth'
    torch.save(net, modelfilename)