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
import os
#% 参数
test_dir='/mnt/data/CCM/snndatabase/TID_TEST_8k_all/'
# test_dir='/mnt/data/CCM/snndatabase/RWCP_test_8k/'
T = 8
# modelpath='/home/handsome/PythonProject/SNN/snnmodel/Cochlea_Coding/T10/Max_num:203_acc:1.0000_val_acc:0.9241.pth'
# modelpath='/home/handsome/PythonProject/SNN/snnmodel/ccm_snn_model_TID_cqt_T10.pth'

#%
# Python语音预加重的代码实现
import numpy as np
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
    Hist = np.histogram(E, bins=10)  # 计算直方图#bins=10
    HistE = Hist[0]
    X_E = Hist[1]
    MaximaE, countMaximaE = findMaxima(HistE, 3)  # 寻找直方图的局部最大值  #3
    if len(MaximaE) >= 2:  # 如果找到了两个以上局部最大值
        T_E = (Weight * X_E[MaximaE[0][0]] + X_E[MaximaE[1][0]]) / (Weight + 1)
    else:
        T_E = E_mean / 2

    # 寻找谱质心的阈值
    Hist = np.histogram(C, bins=10) #bins=10
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3) #3
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




def awgn(x, snr, out='signal', method='vectorized', axis=0):

    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)

    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))

    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)

    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    # Signal power, in dB
    Psdb = 10 * np.log10(Ps)

    # Noise level necessary
    Pn = Psdb - snr

    # Noise vector (or matrix)
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)

    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n
Noise, sr_noise = librosa.load('/mnt/data/CCM/snndatabase/noise92/babble.wav', sr=8000)
def add_noise(sound,snr,noise=Noise):
    min_length = min(len(sound), len(noise))
    start_index = np.random.randint(0, len(noise) - len(sound) + 1)
    sound = sound[:min_length]
    # noise = noise[:min_length]
    noise = noise[start_index:start_index + len(sound)]
    signal_energy = np.sum(sound ** 2)
    noise_energy = np.sum(noise ** 2)

    # 计算混合比例以达到特定SNR
    target_snr_db = snr  # 目标SNR为10dB
    target_snr_linear = 10 ** (target_snr_db / 10)  # 将dB转换为线性比例
    mixing_ratio = np.sqrt(signal_energy / (target_snr_linear * noise_energy))

    # 将声音和噪声按照混合比例混合
    mixed_audio = sound + mixing_ratio * noise
    return mixed_audio

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
            # sounddata=add_noise(sounddata,20)
            sounddata=pre_emphasis(sounddata)

            segments = VAD(sounddata, 8000)

            if len(segments) != 0:
                sounddata = sounddata[segments[0][0]:segments[-1][1]]

            sounddata = normalize_audio_peak(sounddata, 1)

            # sounddata = add_noise(sounddata, 0)
            cqtpec = cqt(sounddata, sr=sample_rate, fmin=32, n_bins=83, hop_length=96)
            # Engry = abs(cqtpec)**2
            cqtm, phase = librosa.core.magphase(cqtpec)
            Engry = cqtm
            # Engry = librosa.amplitude_to_db(cqtm, ref=np.max)

            Min_data = np.min(Engry)
            Max_data = np.max(Engry)
            data_per_file = (Engry - Min_data) / (Max_data - Min_data)

            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
            # print(file_nums_count, '\n')
        folder_nums += 1
    # data = np.array(data)9
    return data, labels


# test_dir = r'D:\浙大项目\SNN\Database\DATA\DATA_WAV'
# test_dir=r'D:\浙大项目\SNN\TID_TEST_8k'.

test_data, test_labels = read_origindata(test_dir)

# Max_num=0;
# LENTH=[]
# for idx in range(len(test_data)):
#     temp=test_data[idx].shape[1];
#     LENTH.append(temp)
#     if temp>Max_num:
#         Max_num=temp
#
# TEST=[]
# for idx in range(len(test_data)):
#    zeronum=abs(Max_num-test_data[idx].shape[1]);
#    temp=np.pad(test_data[idx],((0,0),(0,zeronum)))
#    TEST.append(temp)
# #%%
# plt.figure()
# plt.plot(LENTH)
# plt.show()
# #%%
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(test_data[1], sr=8000, x_axis='time', y_axis='cqt_note')
# plt.colorbar(label='Amplitude (dB)')
# plt.title('CQT spectrogram')
# plt.tight_layout()
# plt.show()

#%

# % 补长
Max_num = 83
TEST = []
for idx in range(len(test_data)):
    if Max_num > test_data[idx].shape[1]:
        zeronum = abs(Max_num - test_data[idx].shape[1]);
        temp = np.pad(test_data[idx], ((0, 0), (0, zeronum)))
    else:
        temp = test_data[idx][:, 0:Max_num]
    TEST.append(temp)

# %
# test=np.array(TEST)
test = np.stack(TEST, axis=0)
test_data = test
#%
# % 参数  平均频率
freq = []
for idx in range(7):
    # temp=np.arange(2**(5+idx),2**(6+idx),((2**(6+idx)-2**(5+idx))/13));
    temp = np.linspace(2 ** (5 + idx), 2 ** (6 + idx), 13);
    freq.append(temp)

freq = np.stack(freq, axis=0)
freq = freq.reshape(-1)
freq = np.unique(freq)
freq = freq[0:84]
Freq = freq[0:test_data.shape[1]]

def lzhikevich_model(T,I, a, b, c, d):
    V_Statue=[]
    v = -65  # 初始化膜电位
    u = b * v
    Timepoint=[]

    spikes = 0
    for idx in range(T):  # 模拟1秒钟（1000毫秒）
        # v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + current)  # 使用Euler方法更新v
        v +=  (0.04 * v ** 2 + 5 * v + 140 - u + I)
        u += a * (b * v - u)  # 使用Euler方法更新u

        if v >= 30:  # 如果膜电位超过阈值，则发放脉冲
            v = c  # 重置膜电位
            u += d  # 更新恢复变量
            Timepoint.append(idx)
        V_Statue.append(v)
    # plt.figure()
    # plt.plot(np.array(V_Statue))
    # plt.show()
    return np.array(Timepoint),np.array(V_Statue)
#
# import numpy as np
file='/mnt/data/CCM/snndatabase/CQT_a_list.txt'
a_list = np.zeros((83, ))
f = open(file, 'r')
content = f.readlines()
f.close()

# a_list=np.array(content)
row=0
for items in content:
    data_i = items.split()
    for x in data_i:
        a_list[row] = x
        row+=1


#% 只做了a的优化
# T = 20
TEST=[]
for idx in range(test_data.shape[0]):
    data_per_file = test_data[idx, :, :]
    F_data = []
    T_data = []
    for time in range(data_per_file.shape[1]):
        T_temp = []
        for frq in range(data_per_file.shape[0]):
            Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[0], 0.25, -65, 8)

            spike_array = np.zeros(T, dtype=bool)
            for i in Timepoint:
                spike_array[i] = True
            T_temp = F_data.append(spike_array)
            # T_temp=np.stack(F_data,axis=0)
            pass
        T_temp = np.stack(F_data, axis=0)
        # T_temp=np.array(T_temp)
        T_data.append(T_temp)
        F_data = []
        T_temp = []
    test=np.stack(T_data, axis=0)
    TEST.append(test)
    print(idx)
test123=np.stack(TEST, axis=0)
# test_data=test123
Test_data=test123

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
#%
modelname=findfile('/home/handsome/PythonProject/SNN/snnmodel/Model_TID_R_MSE/Nobound_CQT_Coding_Eng_low/T8_2/','.pth')
# modelpath='/home/handsome/PythonProject/SNN/snnmodel/Resultmodel_VAD_big_RWCP/Cochlea_Coding_Eng/T10/Max_num:83_acc:1.0000_val_acc:1.0000.pth'
# modelpath='/home/handsome/PythonProject/SNN/snnmodel/Resultmodel_VAD_big/Cochlea_Coding_Eng/T20/Max_num:83_acc:1.0000_val_acc:0.9420.pth'
# savepath='/home/handsome/PythonProject/SNN/snnmodel/Resultmodel_VAD_big/Cochlea_Coding_Eng/T20/figure_T20/'
for idx in range(len(modelname)):
    # modelpath='/home/handsome/PythonProject/SNN/snnmodel/Resultmodel_Nobound_CQT_Encoding_RWCP/T20/Max_num:83_acc:1.0000_val_acc:1.0000_1699328774.pth'
    modelpath=modelname[idx]
    savepath='/home/handsome/PythonProject/SNN/snnmodel/Model_TID_R_MSE/Nobound_CQT_Coding_Eng_low/T8_2/figure_T8/'

    net = torch.load(modelpath, map_location='cpu')
    net.eval()
    with torch.inference_mode():
        SUM = 0
        pred = []
        # T=15
        for idx in range(Test_data.shape[0]):
            test_output = 0
            result1 = Test_data[idx,:, :, :]
            result1 = result1[np.newaxis, :]
            result1=np.array(result1)
            result1 = torch.from_numpy(result1).type(torch.FloatTensor)
            result1 = result1.permute(3, 0, 2, 1)

            for t in range(T):
                # encoded_data=test_data[:,t,:,:]
                # encoded_data = encoder(test_data)
                # test_output += net(encoded_data)
                temp=result1[t]
                # temp=temp[np.newaxis,:,:]
                # temp=torch.from_numpy(temp).type(torch.FloatTensor)
                test_output += net(temp)
            test_output = test_output / T

            Label = np.argmax(test_output.numpy())
            pred.append(Label)
            if Label == test_labels[idx]:
                SUM = SUM + 1
            print(idx)
        print(SUM /Test_data.shape[0])
    #%
    if os.path.exists(savepath):
        print('ready exist')
    else:
        print('ok I make it')
        os.makedirs(savepath)
    figurename=str.split(str.split(modelpath,'/')[-1],'.pth')[0]
    pred_label=np.array(pred)
    true_label=np.array(test_labels)
    from sklearn.metrics import confusion_matrix
    # def confusion(y_label,y_pred):
    con_mat=confusion_matrix(true_label.astype(str),pred_label.astype(str))
    #print(con_mat)
    classes=['0','1','2','3','4','5','6','7','8','9','10']
    #classes.sort()
    plt.imshow(con_mat,cmap=plt.cm.Blues)
    indices=range(len(con_mat))
    plt.xticks(indices,classes)
    plt.yticks(indices,classes)
    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index,second_index,con_mat[second_index][first_index],va='center',ha='center')
    plt.savefig(savepath+figurename+'_ACC:'+str(SUM /Test_data.shape[0])+'_1'+'.tif', dpi=300)
    plt.show()




    # confusion(true_label,pred_label)
    #%
    #%
    cm = confusion_matrix(true_label, pred_label)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(np.arange(11))
    plt.yticks(np.arange(11))
    plt.savefig(savepath+figurename+'_ACC:'+str(SUM /Test_data.shape[0])+'_2'+'.tif', dpi=300)
    plt.show()
