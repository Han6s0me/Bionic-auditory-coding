# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:27:21 2022

@author: CCM
"""
#%%
import wave 
def pcm2wav(pcm_file, wav_file, channels=1, bits=16, sample_rate=16000):
    # 打开 PCM 文件
    pcmf = open(pcm_file, 'rb')
    pcmdata = pcmf.read()
    pcmf.close()
 
    # 打开将要写入的 WAVE 文件
    wavfile = wave.open(wav_file, 'wb')
    # 设置声道数
    wavfile.setnchannels(channels)
    # 设置采样位宽
    wavfile.setsampwidth(bits // 8)
    # 设置采样率
    wavfile.setframerate(sample_rate)
    # 写入 data 部分
    wavfile.writeframes(pcmdata)
    wavfile.close()
 
# pcm2wav(r"C:\Users\CCM\Desktop\3.pcm", r"C:\Users\CCM\Desktop\20230916.wav")
pcm2wav(r"\\192.168.50.140\mic\vvui\audio\2.pcm", r"C:\Users\CCM\Desktop\20230916.wav")

#%%
from libsmop import *
data,FS=audioread(r"C:\Users\CCM\Desktop\20230916.wav")
#%%
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.linspace(0, 6 * np.pi, 100)
data = np.sin(x) + np.random.normal(0, 0.1, x.shape)

peaks, _ = signal.find_peaks(data)
plt.plot(x, data)
plt.plot(x[peaks], data[peaks], "x")
plt.show()
#%%
def lzhikevichModel_New(T,I,a,b,c,d):
    v = -65;                 # %初始膜电位
    u = 0;                   #   %初始恢复变量

    v_thresh = 30;            #   %电位阈值

    step = 1;              # %步长 ms
    
    t = 0;                  #    %计时器

    n = 0;                  #    %步数
    v_states=np.zeros(shape=(T))
    u_states=np.zeros(shape=(T))
    v_del_states=np.zeros(shape=(T))
    u_del_states=np.zeros(shape=(T))
    Time=np.zeros(shape=(T))
    v_states[n] = v    # %状态量初始化
    u_states[n] = u # %输入初始化
    
   # % a = 0.001;                  %模型参数
   # % b = 0.25;                  %模型参数
    #% c = -65;                  %模型参数
    #% d = 8;                  %模型参数
    
    while (t < T-1):
        v_del_states[n] = 0.04 * v_states[n] * v_states[n] + 5 * v_states[n] +140 - u_states[n] + (I/1);            #%状态变化量
        u_del_states[n] = a * (b * (v_states[n]) - u_states[n]);     #%状态量初始化
        
        v_states[n+1] = v_states[n] + v_del_states[n];        #   %更新 赋值 old 和del一个行向量一个列向量导致出来是矩阵
        u_states[n+1] = u_states[n] + u_del_states[n];        #   %控制量
        if v_states[n+1] > v_thresh:        
            v_states[n] = v_thresh
            v_states[n+1] = c;  # % 恢
            u_states[n+1] = u_states[n+1] + d;     # % 恢复

      #  %  时间更新
        Time[n] = t  # %时间记录
        t = t + step  #%仿真步长
        n = n + 1

    [maxv,maxl] = signal.find_peaks(v_states,threshold=29);
    return v_states,maxl,maxv
    # maxl=maxl'
    # maxv=maxv'

# v_states,maxl,maxv=lzhikevichModel_New(0.1,1.45,0.26,-65,2);
v_states,maxl,maxv=lzhikevichModel_New(100,10,0.021,0.25,-65,8);
plt.figure()
plt.plot(v_states)
plt.plot(maxv,v_states[maxv],"x")
plt.show()
#%%
a=[]
for i in range(5):
    a.append(maxv+i)
b=np.array(a)
c=b.reshape(1,-1)

#%%
def normalize_audio_peak(audio, target_peak):
    # 读取音频文件

    # 计算音频的当前峰值
    current_peak = max(abs(audio))

    # 计算缩放系数
    scale = target_peak / current_peak

    # 对音频应用缩放系数
    normalized_audio = audio * scale

    return normalized_audio
#%%
import torch
import torchaudio
import matplotlib.pyplot as plt

filename = r"D:\浙大项目\SNN\Database\DATA\DATA_WAV\3_three_wav\Bmr008-me011-c2-u1-6-1889-3.wav"
waveform, sample_rate = torchaudio.load(filename)
print("Shape of waveform:{}".format(waveform.size())) #音频大小
print("sample rate of waveform:{}".format(sample_rate))#采样率
plt.figure(1)
plt.plot(waveform.t().numpy())
plt.show()
#%%
from scipy import signal
import numpy as np
import math
data=waveform.t().numpy()
data=data.reshape(-1,)

# data=normalize_audio_peak(data,1)
plt.figure()
plt.plot(data)
plt.show()
# low_cutoff = 0.1  # 低截止频率
# high_cutoff = 3000.0  # 高截止频率
# order = 10  # 滤波器阶数

# # 计算归一化截止频率
# low = low_cutoff / (sample_rate / 2)
# high = high_cutoff / (sample_rate / 2)

# # 创建低通滤波器
# b, a = butter(order, high, btype='lowpass')

# # 进行滤波
# data = lfilter(b, a, data)







f, t, Sxx = signal.spectrogram(data, sample_rate)
Engry=10*np.log10(Sxx)
plt.pcolormesh(t, f, Engry, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.figure()
Min_data=np.min(Engry)
Max_data=np.max(Engry)
result1=(Engry-Min_data)/(Max_data-Min_data)
plt.pcolormesh(t, f, result1, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]_New')
plt.show()
# MAX=max(max(Engry));
# result1=(result-MIN)./(MAX-MIN);
#%%
import torch
from spikingjelly.activation_based import encoding
from spikingjelly import visualizing

pe = encoding.PoissonEncoder()

# 仿真20个时间步长，将图像编码为脉冲矩阵并输出
w, h = result1.shape
result1 = torch.from_numpy(result1)
out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
T = 20
for t in range(T):
    out_spike[t] = pe(result1)
plt.plot(out_spike.float().numpy(), 4, 5, 30, 'PoissonEncoder')
plt.axis('off')
plt.show()
#%%
import torch
from spikingjelly.activation_based import encoding
import os
def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # 如果是文件夹，则递归
        if os.path.isdir(file_path):
            findfile(file_path, file_last_name)
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name

def read_origindata(dir):#读取初始编码
    # 读取数据
    folder_nums = 0  # 从第0个文件夹开始，遍历到第3个，共4个文件夹
    file_nums_count = 0  # 文件计数器
    data = []  # 总的训练集集合
    labels = []  # 创建每组数据对应的标签
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        print(folder_nums, folder, folder_path)  # 3 3_腐败水果 ./data/txt_data_for_food/spiking_train/3_腐败水果
        file_path = findfile(folder_path, '.wav')
        label = folder.split('_')[0]
        for file in file_path:
            # print(file)
            # 补零数不影响训练结果，矩阵的第二维大于单个神经元的最大脉冲数量即可
            
            # data_per_file = np.zeros((5, 1000))
            # f = open(file, 'r')
            # content = f.readlines()
            # f.close()
            # # print(len(content)):6
            # row = 0
            # for items in content:
            #     data_i = items.split()
            #     # print(data_i)
            #     col = 0
            #     for x in data_i:
            #         data_per_file[row][col] = x
            #         col += 1
            #     row += 1
            # print(data_per_file)
            
            waveform, sample_rate = torchaudio.load(file)
            sounddata=waveform.t().numpy()
            sounddata=sounddata.reshape(-1,)
            
            
            # target_freq = 48000 
            # # 随机生成原始信号
            # # 计算采样周期
            # T = 1 / sample_rate 
            # # 计算新采样周期
            # new_T = 1 / target_freq 
            # # 计算信号时间长度
            # duration = T * len(sounddata)
            
            # sounddata=signal.resample(sounddata,int(duration / new_T))
            segments = VAD(sounddata, 8000)

            if len(segments)!=0:
                sounddata=sounddata[segments[0][0]:segments[0][1]]

            sounddata=normalize_audio_peak(sounddata,1)
            # f, t, Sxx = signal.spectrogram(sounddata, sample_rate)
            # Engry=10*np.log10(Sxx)
            sounddata=torch.from_numpy(sounddata).type(torch.FloatTensor)
            x1=torch.stft(sounddata,n_fft=128,hop_length=96,win_length=128,window=torch.hann_window(128),onesided=True)
            ttt=torch.sqrt(torch.sum(x1**2,dim=-1))
            Engry=ttt.numpy()
            Min_data=np.min(Engry)
            Max_data=np.max(Engry)
            data_per_file=(Engry-Min_data)/(Max_data-Min_data)
            #pe = encoding.PoissonEncoder()

            # 仿真20个时间步长，将图像编码为脉冲矩阵并输出
            #w, h = result1.shape
            #result1 = torch.from_numpy(result1)
            #out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
            #T = 20
            #for t in range(T):
               # out_spike[t] = pe(result1)
            
            
            #data_per_file=out_spike.float().numpy()
            
            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
            # print(file_nums_count, '\n')
        folder_nums += 1
    #data = np.array(data)
        
    labels = np.array(labels)
    return data, labels


train_dir=r'D:\浙大项目\SNN\TID_DATA_8k'
train_data, train_labels = read_origindata(train_dir)
#%%
Max_num=0;
for idx in range(len(train_data)):
    temp=train_data[idx].shape[1];
    if temp>Max_num:
        Max_num=temp

TEST=[]
for idx in range(len(train_data)):
   zeronum=abs(Max_num-train_data[idx].shape[1]);
   temp=np.pad(train_data[idx],((0,0),(0,zeronum)))
   TEST.append(temp)

#test=np.array(TEST)
test=np.stack(TEST, axis=0)
train_data=test
#%%
a=np.ones(shape=(4,2))
a=np.pad(a,((0,0),(0,3)))

#%%
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(train_data, train_labels, test_size=0.1, random_state=25)
#%%
batch_size=64
#包装数据
def data_package(train_X, train_Y, test_X, test_Y, batch_size):
    # numpy数据转tensor
    x_train_data=[]
  
    x_train_data = torch.from_numpy(train_X).type(torch.FloatTensor)
    y_train_data = torch.from_numpy(train_Y).type(torch.FloatTensor)
    x_test_data=[]

    x_test_data = torch.from_numpy(test_X).type(torch.FloatTensor)
    y_test_data = torch.from_numpy(test_Y).type(torch.FloatTensor)
    # print(x_train_data.size(), y_train_data.size(), x_test_data.size(), y_test_data.size())

    # 将数据tensor和标签tensor包装成Dataset类
    train_dataset = TensorDataset(x_train_data, y_train_data)
    test_dataset = TensorDataset(x_test_data, y_test_data)

    # 将dataset传入DataLoader中，shuffle使得每个epoch中的样本生成顺序不一样，也就是不依次取样本
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


train_loader, test_loader = data_package(X_train, Y_train,X_test, Y_test, batch_size)
#%%
# 订单号补齐
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer



class SNNforauditory(nn.Module):
    def __init__(self, tau):
        super().__init__()


        self.fc = nn.Sequential(
            nn.Flatten(),
            # layer.Dropout(0.7),
            nn.Linear(65*75, 512, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            
            # layer.Dropout(0.7),
            nn.Linear(512, 256, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Dropout(0.5),

        )
        self.fc1 = nn.Sequential(
            # layer.Dropout(0.7),
            nn.Linear(256, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

        )

    def forward(self, x: torch.Tensor):
        x=self.fc(x)
        x = self.fc1(x)
        return x

#%%
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Total GPU numbers:' + str(torch.cuda.device_count()), '\n', 'Being uesd GPU:' + str(device))
torch.cuda.manual_seed(0)
batch_size = 64
# 学习率
learning_rate = 1e-2
# 仿真时长，T 越大，仿真占用显存越大
T =20
# LIF神经元的时间常数tau，tau 越大，拟合速度越慢
tau = 4.0
# 训练轮数，经实验，一般需要设置为 10*tau 以上
train_epoch = 200
# 保存tensorboard日志文件的位置

net = SNNforauditory(tau=tau).to(device)
log_dir=''
# 使用Adam优化器
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,weight_decay=0.001)
train_times = 0
max_test_accuracy = -1

train_accuracy_all = []
train_loss_all = []

validation_accuracy_all = []
validation_loss_all = []
encoder = encoding.PoissonEncoder()
for epoch in range(train_epoch):
    print('------------Epoch:%s------------' % (epoch + 1))
    # 训练模型
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
        # data = data.cuda()
        # label = label.long().cuda()

        # label_one_hot = F.one_hot(label, 10).float()
        label_one_hot = F.one_hot(label.unsqueeze(0).to(torch.int64), 10).float()
       # data_seq = data.permute(2, 0, 1)

        out_spikes_counter_frequency =0
        #w, h = data.shape
        #data = torch.from_numpy(data)
        #out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
            #T = 20
            #for t in range(T):
               # out_spike[t] = pe(result1)
            
            
            #data_per_file=out_spike.float().numpy()
        print(data.shape)
        for t in range(T):
            encoded_data = encoder(data)
            out_spikes_counter_frequency += net(encoded_data)
            #out_spikes_counter_frequency+=net(data_seq[t])
        out_spikes_counter_frequency=out_spikes_counter_frequency/T


        # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
        # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0

        loss= F.mse_loss(out_spikes_counter_frequency, label_one_hot)
        # loss=F.cross_entropy(out_spikes_counter_frequency, label_one_hot)
        loss.backward()
        optimizer.step()
        # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
        functional.reset_net(net)
        train_losses.append(loss.item())

        # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
        train_correct_sum += ( out_spikes_counter_frequency.max(1)[1] == label).float().sum().item()
        train_sum += label.numel()
        train_accuracy = train_correct_sum / train_sum

        if train_times % 1 == 0:
            print('train_times:', train_times, 'train_loss:', '%.5f' % (loss.item()), 'train_accuracy:',
                  '%.5f' % train_accuracy)


        epoch_train_accuracy.append(train_accuracy)
    epoch_train_loss = np.array(train_losses).mean()
    train_loss_all.append(epoch_train_loss)
    train_accuracy_all.append(epoch_train_accuracy[-1])

    # 验证模型
    net.eval()
    with torch.no_grad():
        # 每遍历一次全部数据集，就在测试集上测试一次
        test_sum = 0
        correct_sum = 0
        val_losses = []
        for test_data, test_label in test_loader:
            test_data = test_data.to(device)
            test_label = test_label.long().to(device)
            # test_label_one_hot = F.one_hot(test_label,10).float()
            test_label_one_hot = F.one_hot(test_label.unsqueeze(0).to(torch.int64),10).float()
            # test_data_seq = test_data.permute(2, 0, 1)

            test_output = 0
            # for t in range(T):
            #     test_output +=net(test_data_seq[t])  # batch_size x 4 tensor
            # test_output = test_output / T
            for t in range(T):
                encoded_data = encoder(test_data)
                test_output += net(encoded_data)
            test_output = test_output / T

            val_loss = F.mse_loss(test_output, test_label_one_hot)
            # val_loss = F.cross_entropy(test_output, test_label_one_hot)
            val_losses.append(val_loss.item())
            correct_sum += (test_output.max(1)[1] == test_label).float().sum().item()  # 预测正确的样本数
            test_sum += test_label.numel()  # numel()用来返回数组中元素的个数
            # print('test sample numbers:', test_sum)
            functional.reset_net(net)

        epoch_val_loss = np.array(val_losses).mean()
        validation_loss_all.append(epoch_val_loss)
        test_accuracy = correct_sum / test_sum

        print('Epoch %s' % (epoch + 1), 'test_loss:', '%.5f' % epoch_val_loss, 'test_accuracy:',
              '%.5f' % test_accuracy)

        validation_accuracy_all.append(test_accuracy)

        if max_test_accuracy < test_accuracy:
            max_test_accuracy = test_accuracy
            # print('saving net...')
            # torch.save(net, './models/ccm_snn_model_01.pth')
            # print('saved model successfully')

    print(
        'dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
             train_dir, batch_size, learning_rate, T, log_dir, max_test_accuracy, train_times))
#%%
epochs = range(1, len(validation_accuracy_all) + 1)
plt.figure()
plt.plot(epochs, train_accuracy_all, 'b', label='Training accuracy')
plt.plot(epochs, validation_accuracy_all, 'r', label='Testing accuracy')
plt.title('Training and testing accuracy')
plt.legend()
plt.show()
# plt.savefig('./figures/ccm_snn_acc_2.tif', dpi=300)
plt.figure()
plt.plot(epochs, train_loss_all, 'b', label='Training loss')
plt.plot(epochs, validation_loss_all, 'r', label='Testing loss')

plt.title('Training and testing loss')
plt.legend()
# plt.savefig('./figures/ccm_snn_loss_2.tif', dpi=300)
plt.show()
#%%

torch.save(net,'./ccm_snn_model_06.pth')
#%%
import torch
import torchaudio
from scipy import signal
from spikingjelly.activation_based import encoding
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
net = torch.load('./ccm_snn_model_06.pth',map_location='cpu')
filename = r"D:\浙大项目\SNN\Database\DATA\DATA_WAV\7_seven_wav\Bro004-me013-c2-u1-6-2028-7.wav"
# filename = r"D:\浙大项目\四模态机器人\testsound\录音\7.wav"
waveform, sample_rate = torchaudio.load(filename)

data=waveform.t().numpy()
data=data.reshape(-1,)
plt.figure()
plt.plot(data)
plt.show()
#%%
low_cutoff = 0.1  # 低截止频率
high_cutoff = 3000.0  # 高截止频率
order = 20  # 滤波器阶数

# 计算归一化截止频率
low = low_cutoff / (sample_rate / 2)
high = high_cutoff / (sample_rate / 2)

# 创建低通滤波器
b, a = butter(order, high, btype='lowpass')

# 进行滤波
filtered_data = lfilter(b, a, data)
#%%
data=filtered_data

#%%
# 原始采样率为8000Hz
# sample_rate = 48000 
# 目标采样率为4000Hz
target_freq = 16000 
# 随机生成原始信号
# 计算采样周期
T = 1 / sample_rate 
# 计算新采样周期
new_T = 1 / target_freq 
# 计算信号时间长度
duration = T * len(data)

data=signal.resample(data,int(duration / new_T))
# data=signal.resample(data,int(data.size / 6))

plt.figure()
plt.plot(data)
plt.show()

#%%
# data=normalize_audio_peak(data,1)
segments = VAD(data, 8000)

if len(segments)!=0:
    data=data[segments[0][0]:segments[0][1]]

# data=normalize_audio_peak(data,1)
plt.figure()
plt.plot(data)
plt.show()
#%%
# f, t, Sxx = signal.spectrogram(data,nperseg=256,noverlap=32,nfft=256,fs=target_freq)
data=normalize_audio_peak(data,1)
plt.figure()
plt.plot(data)
plt.show()


t=torch.from_numpy(data).type(torch.FloatTensor)
x1=torch.stft(t,n_fft=128,hop_length=96,win_length=128,window=torch.hann_window(128),onesided=True)
ttt=torch.sqrt(torch.sum(x1**2,dim=-1)) 

f=np.linspace(0,8000,128//2+1)
t=np.linspace(0,data.size/16000,(data.size//96)+1)
# Engry=10*np.log10(Sxx)
Engry=ttt.numpy()

Min_data=np.min(Engry)
Max_data=np.max(Engry)
result1=(Engry-Min_data)/(Max_data-Min_data)
plt.pcolormesh(t, f, result1, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]_New')
plt.show()
#%%
Max_num=75
if Max_num>result1.shape[1]:
    zeronum=abs(Max_num-result1.shape[1]);
    result1=np.pad(result1,((0,0),(0,zeronum)))
else:
    result1=result1[:,0:Max_num]

result1=result1[np.newaxis,:]


T=20
test_output=0
result1=torch.from_numpy(result1).type(torch.FloatTensor)
encoder = encoding.PoissonEncoder()
for t in range(T):
    encoded_data = encoder(result1)
    test_output += net(encoded_data)
test_output = test_output / T


Label=np.argmax(test_output.numpy())
print(Label)
#%%

import numpy as np
import sys
from collections import deque
import matplotlib.pyplot as plt
import scipy.signal
import pyaudio
import struct as st

def ShortTimeEnergy(signal, windowLength, step):
    """
    计算短时能量
    Parameters
    ----------
    signal : 原始信号.
    windowLength : 帧长.
    step : 帧移.
    
    Returns
    -------
    E : 每一帧的能量.
    """
    signal = signal / np.max(signal) # 归一化
    curPos = 0
    L = len(signal)
    numOfFrames  = np.asarray(np.floor((L-windowLength)/step) + 1, dtype=int)
    E = np.zeros((numOfFrames, 1))
    for i in range(numOfFrames):
        window = signal[int(curPos):int(curPos+windowLength-1)];
        E[i] = (1/(windowLength)) * np.sum(np.abs(window**2));
        curPos = curPos + step;
    return E

def SpectralCentroid(signal,windowLength, step, fs):
    """
    计算谱质心
    Parameters
    ----------
    signal : 原始信号.
    windowLength : 帧长.
    step : 帧移.
    fs : 采样率.

    Returns
    -------
    C : 每一帧的谱质心.
    """
    signal = signal / np.max(signal) # 归一化
    curPos = 0
    L = len(signal)
    numOfFrames  = np.asarray(np.floor((L - windowLength) / step) + 1, dtype=int)
    H = np.hamming(windowLength)
    m = ((fs / (2 * windowLength)) * np.arange(1, windowLength, 1)).T
    C = np.zeros((numOfFrames, 1))
    for i in range(numOfFrames):
        window = H * (signal[int(curPos) : int(curPos + windowLength)])
        FFT = np.abs(np.fft.fft(window, 2 * int(windowLength)))
        FFT = FFT[1 : windowLength]
        FFT = FFT / np.max(FFT)
        C[i] = np.sum(m * FFT) / np.sum(FFT)
        if np.sum(window**2) < 0.010:
            C[i] = 0.0
        curPos = curPos + step;
    C = C / (fs/2)
    return C

def findMaxima(f, step):
    """
    寻找局部最大值
    Parameters
    ----------
    f : 输入序列.
    step : 搜寻窗长.

    Returns
    -------
    Maxima : 最大值索引 最大值
    countMaxima : 最大值的数量
    """
    ## STEP 1: 寻找最大值
    countMaxima = 0
    Maxima = []
    for i in range(len(f) - step - 1): # 对于序列中的每一个元素:
        if i >= step:
            if (np.mean(f[i - step : i]) < f[i]) and (np.mean(f[i + 1 : i + step + 1]) < f[i]): 
                # IF the current element is larger than its neighbors (2*step window)
                # --> keep maximum:
                countMaxima = countMaxima + 1
                Maxima.append([i, f[i]])
        else:
            if (np.mean(f[0 : i + 1]) <= f[i]) and (np.mean(f[i + 1 : i + step + 1]) < f[i]):
                # IF the current element is larger than its neighbors (2*step window)
                # --> keep maximum:
                countMaxima = countMaxima + 1
                Maxima.append([i, f[i]])

    ## STEP 2: 对最大值进行进一步处理
    MaximaNew = []
    countNewMaxima = 0
    i = 0
    while i < countMaxima:
        # get current maximum:
        
        curMaxima = Maxima[i][0]
        curMavVal = Maxima[i][1]

        tempMax = [Maxima[i][0]]
        tempVals = [Maxima[i][1]]
        i = i + 1

        # search for "neighbourh maxima":
        while (i < countMaxima) and (Maxima[i][0] - tempMax[len(tempMax) - 1] < step / 2):
            
            tempMax.append(Maxima[i][0])
            tempVals.append(Maxima[i][1])
            i = i + 1
            
        MM = np.max(tempVals)
        MI = np.argmax(tempVals) 
        if MM > 0.02 * np.mean(f): # if the current maximum is "large" enough:
            # keep the maximum of all maxima in the region:
            MaximaNew.append([tempMax[MI], f[tempMax[MI]]])
            countNewMaxima = countNewMaxima + 1   # add maxima
    Maxima = MaximaNew
    countMaxima = countNewMaxima
    
    return Maxima, countMaxima

def VAD(signal, fs):
    win = 0.05
    step = 0.05
    Eor = ShortTimeEnergy(signal, int(win * fs), int(step * fs));
    Cor = SpectralCentroid(signal, int(win * fs), int(step * fs), fs);
    E = scipy.signal.medfilt(Eor[:, 0], 5)
    E = scipy.signal.medfilt(E, 5)
    C = scipy.signal.medfilt(Cor[:, 0], 5)
    C = scipy.signal.medfilt(C, 5)
    
    E_mean = np.mean(E);
    Z_mean = np.mean(C);
    Weight = 100 # 阈值估计的参数
    # 寻找短时能量的阈值
    Hist = np.histogram(E, bins=10) # 计算直方图
    HistE = Hist[0]
    X_E = Hist[1]
    MaximaE, countMaximaE = findMaxima(HistE, 3) # 寻找直方图的局部最大值
    if len(MaximaE) >= 2: # 如果找到了两个以上局部最大值
        T_E = (Weight*X_E[MaximaE[0][0]] + X_E[MaximaE[1][0]]) / (Weight + 1)
    else:
        T_E = E_mean / 2
    
    # 寻找谱质心的阈值
    Hist = np.histogram(C, bins=10)
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3)
    if len(MaximaC)>=2:
        T_C = (Weight*X_C[MaximaC[0][0]]+X_C[MaximaC[1][0]]) / (Weight+1)
    else:
        T_C = Z_mean / 2
    
    # 阈值判断
    Flags1 = (E>=T_E)
    Flags2 = (C>=T_C)
    flags = np.array(Flags1 & Flags2, dtype=int)
    
    ## 提取语音片段
    count = 1
    segments = []
    while count < len(flags): # 当还有未处理的帧时
        # 初始化
        curX = []
        countTemp = 1
        while ((flags[count - 1] == 1) and (count < len(flags))):
            if countTemp == 1: # 如果是该语音段的第一帧
                Limit1 = np.round((count-1)*step*fs)+1 # 设置该语音段的开始边界
                if Limit1 < 1:
                    Limit1 = 1
            count = count + 1 		# 计数器加一
            countTemp = countTemp + 1	# 当前语音段的计数器加一
            
        if countTemp > 1: # 如果当前循环中有语音段
            Limit2 = np.round((count - 1) * step * fs) # 设置该语音段的结束边界
            if Limit2 > len(signal):
                Limit2 = len(signal)
            # 将该语音段的首尾位置加入到segments的最后一行
            segments.append([int(Limit1), int(Limit2)])
        count = count + 1
        
    # 合并重叠的语音段
    for i in range(len(segments) - 1): # 对每一个语音段进行处理
        if segments[i][1] >= segments[i + 1][0]:
            segments[i][1] = segments[i + 1][1]
            segments[i + 1, :] = []
            i = 1

    return segments
#%%
signal=data
segments = VAD(signal, 8000) # 端点检测
index = 0
for seg in segments:
    if index < seg[0]:
        x = np.linspace(index, seg[0], seg[0] - index, endpoint=True, dtype=int)
        y = signal[index:seg[0]]
        plt.plot(x, y, 'g', alpha=1)
    x = np.linspace(seg[0], seg[1], seg[1] - seg[0], endpoint=True, dtype=int)
    y = signal[seg[0]:seg[1]]
    plt.plot(x, y, 'r', alpha=1)
    index = seg[1]            
x = np.linspace(index, len(signal), len(signal) - index, endpoint=True, dtype=int)
y = signal[index:len(signal)]
plt.plot(x, y, 'g', alpha=1)
# plt.ylim((-32768, 32767))
plt.show()

#%%

import torch
from spikingjelly.activation_based import encoding
import os
def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # 如果是文件夹，则递归
        if os.path.isdir(file_path):
            findfile(file_path, file_last_name)
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name

def read_origindata(dir):#读取初始编码
    # 读取数据
    folder_nums = 0  # 从第0个文件夹开始，遍历到第3个，共4个文件夹
    file_nums_count = 0  # 文件计数器
    data = []  # 总的训练集集合
    labels = []  # 创建每组数据对应的标签
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        print(folder_nums, folder, folder_path)  # 3 3_腐败水果 ./data/txt_data_for_food/spiking_train/3_腐败水果
        file_path = findfile(folder_path, '.wav')
        label = folder.split('_')[0]
        for file in file_path:
            # print(file)
            # 补零数不影响训练结果，矩阵的第二维大于单个神经元的最大脉冲数量即可
            
            # data_per_file = np.zeros((5, 1000))
            # f = open(file, 'r')
            # content = f.readlines()
            # f.close()
            # # print(len(content)):6
            # row = 0
            # for items in content:
            #     data_i = items.split()
            #     # print(data_i)
            #     col = 0
            #     for x in data_i:
            #         data_per_file[row][col] = x
            #         col += 1
            #     row += 1
            # print(data_per_file)
            
            waveform, sample_rate = torchaudio.load(file)
            sounddata=waveform.t().numpy()
            sounddata=sounddata.reshape(-1,)
            
            
            # target_freq = 48000 
            # # 随机生成原始信号
            # # 计算采样周期
            # T = 1 / sample_rate 
            # # 计算新采样周期
            # new_T = 1 / target_freq 
            # # 计算信号时间长度
            # duration = T * len(sounddata)
            
            # sounddata=signal.resample(sounddata,int(duration / new_T))
            segments = VAD(sounddata, 8000)

            if len(segments)!=0:
                sounddata=sounddata[segments[0][0]:segments[0][1]]

            sounddata=normalize_audio_peak(sounddata,1)
            # f, t, Sxx = signal.spectrogram(sounddata, sample_rate)
            # Engry=10*np.log10(Sxx)
            sounddata=torch.from_numpy(sounddata).type(torch.FloatTensor)
            x1=torch.stft(sounddata,n_fft=128,hop_length=96,win_length=128,window=torch.hann_window(128),onesided=True)
            ttt=torch.sqrt(torch.sum(x1**2,dim=-1))
            Engry=ttt.numpy()
            Min_data=np.min(Engry)
            Max_data=np.max(Engry)
            data_per_file=(Engry-Min_data)/(Max_data-Min_data)
            #pe = encoding.PoissonEncoder()

            # 仿真20个时间步长，将图像编码为脉冲矩阵并输出
            #w, h = result1.shape
            #result1 = torch.from_numpy(result1)
            #out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
            #T = 20
            #for t in range(T):
               # out_spike[t] = pe(result1)
            
            
            #data_per_file=out_spike.float().numpy()
            
            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
            # print(file_nums_count, '\n')
        folder_nums += 1
    #data = np.array(data)
        
    labels = np.array(labels)
    return data, labels


test_dir=r'D:\浙大项目\SNN\Database\DATA\DATA_WAV'
test_data, test_labels = read_origindata(test_dir)
#%%
Max_num=75
TEST=[]
for idx in range(len(test_data)):
   if Max_num>test_data[idx].shape[1]:
       zeronum=abs(Max_num-test_data[idx].shape[1]);
       temp=np.pad(test_data[idx],((0,0),(0,zeronum)))
   else:
       temp=test_data[idx][:,0:Max_num]
   TEST.append(temp)

#%%
#test=np.array(TEST)
test=np.stack(TEST, axis=0)
test_data=test


#%%

T=20

SUM=0
for idx in range(test_data.shape[0]):
    test_output=0
    result1=test_data[idx,:,:]
    result1=result1[np.newaxis,:]
    result1=torch.from_numpy(result1).type(torch.FloatTensor)
    encoder = encoding.PoissonEncoder()
    for t in range(T):
        encoded_data = encoder(result1)
        test_output += net(encoded_data)
    test_output = test_output / T
    
    
    Label=np.argmax(test_output.numpy())
    if Label==test_labels[idx]:
        SUM=SUM+1
    # print(Label)
print(SUM/test_data.shape[0])
#%%
import numpy as np

import matplotlib.pyplot as plt

from librosa import cqt

import torch
import torchaudio
from scipy import signal
from spikingjelly.activation_based import encoding
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

filename = r"D:\浙大项目\SNN\Database\DATA\DATA_WAV\4_four_wav\Bmr003-me028-c2-u1-6-0581-4.wav"
# filename = r"D:\浙大项目\四模态机器人\testsound\录音\7.wav"
waveform, sample_rate = torchaudio.load(filename)
waveform=np.array(waveform)
waveform=waveform.reshape(-1)
#%%
import librosa
# cqtpec = cqt(waveform, sr=sample_rate,fmin=32,n_bins=83,hop_length=256)
cqtpec = cqt(waveform, sr=sample_rate,fmin=32,n_bins=83,hop_length=96)
Engry=np.abs(cqtpec)
# Engry=np.log10(np.abs(cqtpec))

# sounddata=torch.from_numpy(waveform).type(torch.FloatTensor)
# x1=torch.stft(sounddata,n_fft=128,hop_length=96,win_length=128,window=torch.hann_window(128),onesided=True)
# ttt=torch.sqrt(torch.sum(x1**2,dim=-1))
# Engry=ttt.numpy()
# # Engry=10*np.log10(Engry)
# f=np.linspace(0,4000,128//2+1)
# t=np.linspace(0,waveform.size/8000,(waveform.size//96)+1)

# plt.imshow(eng,aspect='auto', origin='lower')

# plt.xlabel('Time Steps')

# plt.ylabel('Freq bins')

Min_data=np.min(Engry)
Max_data=np.max(Engry)
data_per_file=(Engry-Min_data)/(Max_data-Min_data)

# plt.imshow(data_per_file,aspect='auto', origin='lower')
# plt.pcolormesh(t, f, data_per_file, shading='gouraud')
# plt.xlabel('Time Steps')

# plt.ylabel('Freq bins')
# fmin=librosa.note_to_hz('E0')
#%%
import torch
from spikingjelly.activation_based import encoding
from spikingjelly import visualizing

pe = encoding.PoissonEncoder()

# 仿真20个时间步长，将图像编码为脉冲矩阵并输出
w, h = data_per_file.shape
result1 = torch.from_numpy(data_per_file)
out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
T = 20
for t in range(T):
    out_spike[t] = pe(result1)
test=out_spike.numpy()
# plt.plot(out_spike.float().numpy(), 4, 5, 30, 'PoissonEncoder')
# plt.axis('off')
# plt.show()
#%%
def arange(start, stop, step):
    return list(range(start, stop, step))
freq=[]
for idx in range(7):
    # temp=np.arange(2**(5+idx),2**(6+idx),((2**(6+idx)-2**(5+idx))/13));
    # temp=np.linspace(2**(5+idx),2**(6+idx),13);
    temp=np.linspace(2**(5+idx),2**(6+idx),13);
    freq.append(temp)
    
freq=np.stack(freq, axis=0)
freq=freq.reshape(-1)
freq=np.unique(freq)
freq=freq[0:84]
#%%
Freq=freq[0:data_per_file.shape[0]]
T=np.arange(0,data_per_file.shape[1],1)

# plt.imshow(Freq,T,data_per_file,aspect='auto', origin='lower')
plt.pcolormesh(T, Freq, data_per_file, shading='gouraud')

plt.xlabel('Time Steps')

plt.ylabel('Freq bins')
#%%
# Freq=f
Freq_100=np.where(Freq<=100)[0]
Freq_500=np.where((Freq>100) & (Freq<=500))[0]
Freq_1000=np.where((Freq>500) & (Freq<=1000))[0]
Freq_2000=np.where((Freq>1000) & (Freq<=2000))[0]
Freq_3000=np.where((Freq>2000) & (Freq<=3000))[0]
Freq_4000=np.where((Freq>3000))[0]

a6=np.linspace(1.4,1.95,Freq_4000.size);   
b6=np.linspace(0.19,0.185,Freq_4000.size);

a5=np.linspace(0.7,1.4,Freq_3000.size);   

b5=np.linspace(0.21,0.18,Freq_3000.size);

a4=np.linspace(0.25,0.94,Freq_2000.size);   
b4=np.linspace(0.2,0.18,Freq_2000.size);

a3=np.linspace(0.08,0.255,Freq_1000.size);   
b3=np.linspace(0.195,0.18,Freq_1000.size);

a2=np.linspace(0.026,0.062,Freq_500.size);   
b2=np.linspace(0.22,0.2,Freq_500.size);

a1=np.linspace(0.03,0.025,Freq_100.size);   
b1=np.linspace(0.18,0.22,Freq_100.size);

a_arr=np.concatenate((a1,a2,a3,a4,a5,a6))
b_arr=np.concatenate((b1,b2,b3,b4,b5,b6))

#%%
T=20
F_data=[]
T_data=[]
for time in range(data_per_file.shape[1]):
    T_temp=[]
    for frq in range(data_per_file.shape[0]):
        # v_states,maxl,maxv=lzhikevichModel_New(10,data_per_file[frq][time]*100,0.03,0.18,-55,8)
        if frq>=Freq_100[0] & frq<=Freq_100[-1]:
            v_states,maxl,maxv=lzhikevichModel_New(T,data_per_file[frq][time]*1000,a_arr[frq],b_arr[frq],-55,8)
        elif frq>=Freq_500[0] & frq<=Freq_500[-1]:
            v_states,maxl,maxv=lzhikevichModel_New(T,data_per_file[frq][time]*1000,a_arr[frq],b_arr[frq],-55,8)
        elif frq>=Freq_1000[0] & frq<=Freq_1000[-1]:
            v_states,maxl,maxv=lzhikevichModel_New(T,data_per_file[frq][time]*1000,a_arr[frq],b_arr[frq],-55,12)
        elif frq>=Freq_2000[0] & frq<=Freq_2000[-1]:
            v_states,maxl,maxv=lzhikevichModel_New(T,data_per_file[frq][time]*1000,a_arr[frq],b_arr[frq],-55,15)
        elif frq>=Freq_3000[0] & frq<=Freq_3000[-1]:
            v_states,maxl,maxv=lzhikevichModel_New(T,data_per_file[frq][time]*1000,a_arr[frq],b_arr[frq],-65,18)
        elif frq>=Freq_4000[0] & frq<=Freq_4000[-1]:
            v_states,maxl,maxv=lzhikevichModel_New(T,data_per_file[frq][time]*1000,a_arr[frq],b_arr[frq],-55,18)
        spike_array = np.zeros(T, dtype=bool)
        for i in maxv:
            spike_array[i] = True
        T_temp=F_data.append(spike_array)
        # T_temp=np.stack(F_data,axis=0)
        pass
    T_temp=np.stack(F_data,axis=0)
    # T_temp=np.array(T_temp)
    T_data.append(T_temp)
    F_data=[]
    T_temp=[]
        
test=np.stack(T_data, axis=0)
# testtest=np.array(T_data)

#%%
F_data=[]
T_data=[]
for time in range(data_per_file.shape[1]):
    T_temp=[]
    for frq in range(data_per_file.shape[0]):
        v_states,maxl,maxv=lzhikevichModel_New(100,data_per_file[frq][time]*100,0.03,0.18,-55,8)

      
        spike_array = np.zeros(100, dtype=bool)
        for i in maxv:
            spike_array[i] = True
        T_temp=F_data.append(spike_array)
        # T_temp=np.stack(F_data,axis=0)
        pass
    T_temp=np.stack(F_data,axis=0)
    # T_temp=np.array(T_temp)
    T_data.append(T_temp)
    F_data=[]
    T_temp=[]
        
test=np.stack(T_data, axis=0)
#%%
f=[]
t=[]
for T in range(test.shape[0]):
    temp= np.where(test[T,:,:] == True)
    nern=temp[0]
    timestep=temp[1]+20*T
    f.append(nern)
    t.append(timestep)
Freqq=np.concatenate(f,axis=0)
Timee=np.concatenate(t,axis=0)
#%%
plt.figure()
plt.plot(Timee,Freqq,'|', color='k')
plt.show()

#%%
fig = plt.figure()

ax = fig.gca(projection='3d')
for T in range(test.shape[0]):
    t=np.linspace(0+T*100,100+T*100,100)
    f=np.linspace(0,test.shape[1],test.shape[1])
    plt.plot(t,f,test[T,:,:],'|', color='k')
plt.show()
#%%
from brian2 import *

start_scope()

N = 100
tau = 10*ms
eqs = '''
dv/dt = (2-v)/tau : 1
'''

G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='exact')
G.v = 'rand()'

spikemon = SpikeMonitor(G)

run(50*ms)

plot(spikemon.t/ms, spikemon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index');
#%%
import numpy as np
t=spikemon.t/ms
n_count=np.array(spikemon.i)
plot(t, n_count, '.k')
xlabel('Time (ms)')
ylabel('Neuron index');