#%%
from scipy import signal
import numpy as np

import torch
import torchaudio
import matplotlib.pyplot as plt

from spikingjelly.activation_based import encoding
from spikingjelly import visualizing
import os
from librosa import cqt
import librosa
# % 导入数据
import torch
from spikingjelly.activation_based import encoding
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

#% 参数
train_dir = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/'
# train_dir = '/mnt/data/CCM/snndatabase/RWCP_train_8k/'
T=10
batch_size = 64
learning_rate = 1e-2
tau = 4.0
train_epoch = 200
savepath='./snnmodel/Model_TID_R_MSE/Freq_encoding/T2/'
# random_state=10

def pre_emphasis(signal, coefficient=0.97):
    """
    对语音信号进行预加重处理
    :param signal: 输入的语音信号
    :param coefficient: 预加重系数，默认为0.97
    :return: 预加重后的语音信号
    """
    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])
    return emphasized_signal
# % 音频缩放
def normalize_audio_peak(audio, target_peak):
    # 读取音频文件

    # 计算音频的当前峰值
    current_peak = max(abs(audio))

    # 计算缩放系数
    scale = target_peak / current_peak

    # 对音频应用缩放系数
    normalized_audio = audio * scale

    return normalized_audio


# % VAD 语音端点检测


import sys
from collections import deque

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
    signal = signal / np.max(signal)  # 归一化
    curPos = 0
    L = len(signal)
    numOfFrames = np.asarray(np.floor((L - windowLength) / step) + 1, dtype=int)
    E = np.zeros((numOfFrames, 1))
    for i in range(numOfFrames):
        window = signal[int(curPos):int(curPos + windowLength - 1)];
        E[i] = (1 / (windowLength)) * np.sum(np.abs(window ** 2));
        curPos = curPos + step;
    return E


def SpectralCentroid(signal, windowLength, step, fs):
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
    signal = signal / np.max(signal)  # 归一化
    curPos = 0
    L = len(signal)
    numOfFrames = np.asarray(np.floor((L - windowLength) / step) + 1, dtype=int)
    H = np.hamming(windowLength)
    m = ((fs / (2 * windowLength)) * np.arange(1, windowLength, 1)).T
    C = np.zeros((numOfFrames, 1))
    for i in range(numOfFrames):
        window = H * (signal[int(curPos): int(curPos + windowLength)])
        FFT = np.abs(np.fft.fft(window, 2 * int(windowLength)))
        FFT = FFT[1: windowLength]
        FFT = FFT / np.max(FFT)
        C[i] = np.sum(m * FFT) / np.sum(FFT)
        if np.sum(window ** 2) < 0.010:
            C[i] = 0.0
        curPos = curPos + step;
    C = C / (fs / 2)
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
    for i in range(len(f) - step - 1):  # 对于序列中的每一个元素:
        if i >= step:
            if (np.mean(f[i - step: i]) < f[i]) and (np.mean(f[i + 1: i + step + 1]) < f[i]):
                # IF the current element is larger than its neighbors (2*step window)
                # --> keep maximum:
                countMaxima = countMaxima + 1
                Maxima.append([i, f[i]])
        else:
            if (np.mean(f[0: i + 1]) <= f[i]) and (np.mean(f[i + 1: i + step + 1]) < f[i]):
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
        if MM > 0.02 * np.mean(f):  # if the current maximum is "large" enough:
            # keep the maximum of all maxima in the region:
            MaximaNew.append([tempMax[MI], f[tempMax[MI]]])
            countNewMaxima = countNewMaxima + 1  # add maxima
    Maxima = MaximaNew
    countMaxima = countNewMaxima

    return Maxima, countMaxima


def VAD(signal, fs):
    win = 0.05
    step = 0.05
    # win = 0.025
    # step = 0.025
    Eor = ShortTimeEnergy(signal, int(win * fs), int(step * fs));
    Cor = SpectralCentroid(signal, int(win * fs), int(step * fs), fs);
    E = scipy.signal.medfilt(Eor[:, 0], 5)
    E = scipy.signal.medfilt(E, 5)
    C = scipy.signal.medfilt(Cor[:, 0], 5)
    C = scipy.signal.medfilt(C, 5)

    E_mean = np.mean(E);
    Z_mean = np.mean(C);
    Weight = 100  # 阈值估计的参数

    # 寻找短时能量的阈值
    Hist = np.histogram(E, bins=10)  # 计算直方图#bins=10  3
    HistE = Hist[0]
    X_E = Hist[1]
    MaximaE, countMaximaE = findMaxima(HistE, 3)  # 寻找直方图的局部最大值  #3 10
    if len(MaximaE) >= 2:  # 如果找到了两个以上局部最大值
        T_E = (Weight * X_E[MaximaE[0][0]] + X_E[MaximaE[1][0]]) / (Weight + 1)
    else:
        T_E = E_mean / 2

    # 寻找谱质心的阈值
    Hist = np.histogram(C, bins=10) #bins=10 3
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3) #3 10
    if len(MaximaC) >= 2:
        T_C = (Weight * X_C[MaximaC[0][0]] + X_C[MaximaC[1][0]]) / (Weight + 1)
    else:
        T_C = Z_mean / 2

    # 阈值判断
    Flags1 = (E >= T_E)
    Flags2 = (C >= T_C)
    flags = np.array(Flags1 & Flags2, dtype=int)

    ## 提取语音片段
    count = 1
    segments = []
    while count < len(flags):  # 当还有未处理的帧时
        # 初始化
        curX = []
        countTemp = 1
        while ((flags[count - 1] == 1) and (count < len(flags))):
            if countTemp == 1:  # 如果是该语音段的第一帧
                Limit1 = np.round((count - 1) * step * fs) + 1  # 设置该语音段的开始边界
                if Limit1 < 1:
                    Limit1 = 1
            count = count + 1  # 计数器加一
            countTemp = countTemp + 1  # 当前语音段的计数器加一

        if countTemp > 1:  # 如果当前循环中有语音段
            Limit2 = np.round((count - 1) * step * fs)  # 设置该语音段的结束边界
            if Limit2 > len(signal):
                Limit2 = len(signal)
            # 将该语音段的首尾位置加入到segments的最后一行
            segments.append([int(Limit1), int(Limit2)])
        count = count + 1

    # 合并重叠的语音段
    for i in range(len(segments) - 1):  # 对每一个语音段进行处理
        if segments[i][1] >= segments[i + 1][0]:
            segments[i][1] = segments[i + 1][1]
            segments[i + 1, :] = []
            i = 1

    return segments



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


def read_origindata(dir):  # 读取初始编码
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

            waveform, sample_rate = torchaudio.load(file)
            sounddata = waveform.t().numpy()
            sounddata = sounddata.reshape(-1, )
            sounddata = pre_emphasis(sounddata)

            segments = VAD(sounddata, 8000)
            if len(segments) != 0:
                sounddata = sounddata[segments[0][0]:segments[-1][1]]
            sounddata = normalize_audio_peak(sounddata, 1)

            cqtpec = cqt(sounddata, sr=sample_rate, fmin=32, n_bins=83, hop_length=96)
            # Engry = abs(cqtpec)
            cqtm, phase = librosa.core.magphase(cqtpec)
            Engry = cqtm


            Min_data = np.min(Engry)
            Max_data = np.max(Engry)
            data_per_file = (Engry - Min_data) / (Max_data - Min_data)

            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
            # print(file_nums_count, '\n')
        folder_nums += 1
    # data = np.array(data)

    labels = np.array(labels)
    return data, labels


# train_dir=r'D:\浙大项目\SNN\Database\DATA\DATA_WAV'
# train_dir = '/mnt/data/CCM/snndatabase/TID_DATA_8k/'
train_data, train_labels = read_origindata(train_dir)

#% 补成统一长度
Max_num=83
# for idx in range(len(train_data)):
#     temp=train_data[idx].shape[1]
#     if temp>Max_num:
#         Max_num=temp

TEST=[]
for idx in range(len(train_data)):
    if Max_num > train_data[idx].shape[1]:
       zeronum=abs(Max_num-train_data[idx].shape[1])
       temp=np.pad(train_data[idx],((0,0),(0,zeronum)))
       # TEST.append(temp)
    else:
       temp = train_data[idx][:, 0:Max_num]
    TEST.append(temp)

#test=np.array(TEST)
test=np.stack(TEST, axis=0)
train_data=test

#%  如果要在模型中直接泊松编码不要运动这段  泊松编码
encoder = encoding.PoissonEncoder()

# T=10

TEST=[]
for idx in range(train_data.shape[0]):
# for idx in range(1):
    data_per_file = train_data[idx, :, :]
    x = torch.from_numpy(data_per_file)
    w, h = x.shape
    out_spike = torch.full((T, w, h), 0, dtype=torch.bool)
    for t in range(T):
        out_spike[t] = encoder(x)
    TEST.append(np.array(out_spike))
    print(idx)
test123=np.stack(TEST, axis=0)
train_data=test123

#%% 保存数据
from scipy.io import savemat
temp = torch.from_numpy(train_data).type(torch.FloatTensor)
train_data=temp.permute(0,3,2,1)
train_data=np.array(train_data)
train_data = np.squeeze(train_data)
savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Possion_coding/train_data_10T_1.mat', {'train_data': train_data})
savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Possion_coding/label_data_10T_1.mat', {'train_labels': train_labels})
#%%
from scipy.io import loadmat

# 读取MATLAB文件
mat_data = loadmat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Possion_coding/train_data_2T.mat')

# mat_data 是一个字典，其中包含MATLAB文件中的变量
# 你可以通过变量名访问数据，例如：
variable_name = 'train_data'
data = mat_data[variable_name]

# 打印数据
print(data)
#%%
# #%%
#% 数据划分
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
#%
for id in range(10):
    random_state=(id+1)*10
    (X_train, X_test, Y_train, Y_test) = train_test_split(train_data, train_labels, test_size=0.1, random_state=random_state)

    # % 数据封装
    # batch_size = 64


    # 包装数据
    def data_package(train_X, train_Y, test_X, test_Y, batch_size):
        # numpy数据转tensor
        x_train_data = []

        x_train_data = torch.from_numpy(train_X).type(torch.FloatTensor)
        y_train_data = torch.from_numpy(train_Y).type(torch.FloatTensor)
        x_test_data = []

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


    train_loader, test_loader = data_package(X_train, Y_train, X_test, Y_test, batch_size)

    # % 构建网络

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    import torch.nn.functional as F


    class SNNforauditory(nn.Module):
        def __init__(self, tau):
            super().__init__()

            self.fc = nn.Sequential(
                nn.Flatten(),
                # layer.Dropout(0.7),
                # nn.Linear(65*92, 512, bias=False),
                nn.Linear(83 * Max_num, 1024, bias=False),
                neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                layer.Dropout(0.5),

                # layer.Dropout(0.7),
                # nn.Linear(512, 256, bias=False),
                # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                # layer.Dropout(0.5),

            )
            self.fc1 = nn.Sequential(
                # layer.Dropout(0.7),
                nn.Linear(1024, 11, bias=False),
                neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

            )

        def forward(self, x: torch.Tensor):
            x = self.fc(x)
            x = self.fc1(x)
            return x


    # % 网络训练

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print('Total GPU numbers:' + str(torch.cuda.device_count()), '\n', 'Being uesd GPU:' + str(device))
    torch.cuda.manual_seed(0)
    # batch_size = 64
    # 学习率
    # learning_rate = 1e-2
    # # 仿真时长，T 越大，仿真占用显存越大
    # # T = 10
    # # LIF神经元的时间常数tau，tau 越大，拟合速度越慢
    # tau = 4.0
    # # 训练轮数，经实验，一般需要设置为 10*tau 以上
    # train_epoch = 200
    # 保存tensorboard日志文件的位置

    net = SNNforauditory(tau=tau).to(device)
    log_dir = ''
    # 使用Adam优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
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

            label_one_hot = F.one_hot(label, 11).float()
            # label_one_hot = F.one_hot(label.unsqueeze(0).to(torch.int64), 11).float()
            # data_seq = data.permute(2, 0, 1)

            out_spikes_counter_frequency = 0
            data_seq = data.permute(1, 0, 2, 3)  # 如果要在模型中直接泊松编码注释这行
            # w, h = data.shape
            # data = torch.from_numpy(data)
            # out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
            # T = 20
            # for t in range(T):
            # out_spike[t] = pe(result1)

            # data_per_file=out_spike.float().numpy()
            print(data.shape)
            for t in range(T):
                # encoded_data = encoder(data)  #如果要在模型中直接泊松编码 取消注释这行
                # out_spikes_counter_frequency += net(encoded_data)  #如果要在模型中直接泊松编码 取消注释这行
                out_spikes_counter_frequency += net(data_seq[t])  # 如果要在模型中直接泊松编码注释这行
            out_spikes_counter_frequency = out_spikes_counter_frequency / T

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0

            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            # label_one_hot = label_one_hot.squeeze(0)
            # label_one_hot=label
            # loss=F.cross_entropy(out_spikes_counter_frequency, label_one_hot)
            # loss = torch.nn.CrossEntropyLoss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)
            train_losses.append(loss.item())

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
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

        # 验证模型
        net.eval()
        # with torch.no_grad():
        with torch.inference_mode():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            val_losses = []
            for test_data, test_label in test_loader:
                test_data = test_data.to(device)
                test_label = test_label.long().to(device)
                test_label_one_hot = F.one_hot(test_label,11).float()
                # test_label_one_hot = F.one_hot(test_label.unsqueeze(0).to(torch.int64), 11).float()
                # test_data_seq = test_data.permute(2, 0, 1)
                test_data_seq = test_data.permute(1, 0, 2, 3)  # 如果要在模型中直接泊松编码 取消注释这行

                test_output = 0
                # for t in range(T):
                #     test_output +=net(test_data_seq[t])  # batch_size x 4 tensor
                # test_output = test_output / T
                for t in range(T):
                    # encoded_data = encoder(test_data) #如果要在模型中直接泊松编码 取消注释这行
                    # test_output += net(encoded_data) #如果要在模型中直接泊松编码 取消注释这行
                    test_output += net(test_data_seq[t])  # 如果要在模型中直接泊松编码注释这行
                test_output = test_output / T

                val_loss = F.mse_loss(test_output, test_label_one_hot)
                # test_label_one_hot = test_label_one_hot.squeeze(0)
                # test_label_one_hot=test_label
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
        scheduler.step(epoch_val_loss)

        print(
            'dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
                train_dir, batch_size, learning_rate, T, log_dir, max_test_accuracy, train_times))
    # % 可视化训练结果
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

    now = np.datetime64('now')
    timestamp = now.astype('int64')
    # % 保存网络
    epochs=np.linspace(1, len(validation_accuracy_all) , len(validation_accuracy_all) )
    import csv
    # savepath='./snnmodel/Freq_Coding/T10/'
    if os.path.exists(savepath):
        print('ready exist')
    else:
        print('ok I make it')
        os.makedirs(savepath)
    filename=savepath+'Max_num:%d'%Max_num+'_'+'acc:%.4f'%train_accuracy_all[-1]+'_'+'val_acc:%.4f'%validation_accuracy_all[-1]+'_'+str(timestamp)+'_train_result.csv'
    with open(filename, 'w', newline='') as file:
        fieldnames = ['epochs', 'train_accuracy_all','train_loss_all','validation_accuracy_all','validation_loss_all']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入CSV文件的表头
        writer.writeheader()
        # 写入数据
        for epochs, train_accuracy_alls, train_loss_alls,validation_accuracy_alls,validation_loss_alls in zip(epochs, np.array(train_accuracy_all),np.array(train_loss_all),np.array(validation_accuracy_all),np.array(validation_loss_all)):
            writer.writerow({'epochs': epochs, 'train_accuracy_all': train_accuracy_alls,'train_loss_all':train_loss_alls,'validation_accuracy_all':validation_accuracy_alls,'validation_loss_all':validation_loss_alls})

    modelfilename=savepath+'Max_num:%d'%Max_num+'_'+'acc:%.4f'%train_accuracy_all[-1]+'_'+'val_acc:%.4f'%validation_accuracy_all[-1]+'_'+str(timestamp)+'.pth'
    torch.save(net, modelfilename)

    # torch.save(net.state_dict(), './snnmodel/ccm_snn_model_TID_cqt_possion_T5_2.pth')