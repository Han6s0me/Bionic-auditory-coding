# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:43:00 2023

@author: CCM
"""

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
#%%
# filename = r"F:\浙大项目\Auditory_Database\DATA_WAV\4_four_wav\Bmr003-me028-c2-u1-6-0581-4.wav"
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k/3/50.wav'
filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/8_ring/097.wav'
# filename = r"D:\浙大项目\四模态机器人\testsound\录音\7.wav"
waveform, sample_rate = torchaudio.load(filename)
waveform=np.array(waveform)
waveform=waveform.reshape(-1)
#%%
target_sample_rate=8000
original_sample_rate=16000
dtype = np.int16
with open('/mnt/data/CCM/snndatabase/RWCP_train_raw/8_ring/097.raw', 'rb') as file:
    data = np.fromfile(file, dtype=dtype)
downsampled_data_resampled = signal.resample(data, int(len(data) * target_sample_rate / original_sample_rate))
waveform=downsampled_data_resampled
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
    # win = 0.025
    # step = 0.025
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
    Weight = 100# 阈值估计的参数
    # energy = librosa.feature.rms(y=signal)
    # # 计算能量的平均值和标准差
    # energy_mean = np.mean(energy)
    # energy_std = np.std(energy)

    # 设置能量阈值
    # Weight = energy_mean + energy_std
    # print(Weight)
    # 寻找短时能量的阈值
    Hist = np.histogram(E, bins=10)  # 计算直方图
    HistE = Hist[0]
    X_E = Hist[1]
    MaximaE, countMaximaE = findMaxima(HistE, 3)  # 寻找直方图的局部最大值
    if len(MaximaE) >= 2:  # 如果找到了两个以上局部最大值
        T_E = (Weight * X_E[MaximaE[0][0]] + X_E[MaximaE[1][0]]) / (Weight + 1)
    else:
        T_E = E_mean / 2

    # 寻找谱质心的阈值
    Hist = np.histogram(C, bins=10)
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3)
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
#%%
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
#%%
import math
def add_noise(data, SNR, sr=8000):
    #读取语音文件data和fs
    # src, sr = librosa.core.load(audio_path, sr=sr)
    src=data
    #
    random_values = np.random.rand(len(src))
    #计算语音信号功率Ps和噪声功率Pn1
    Ps = np.sum(src ** 2) / len(src)
    Pn1 = np.sum(random_values ** 2) / len(random_values)

    # 计算k值
    k=math.sqrt(Ps/(10**(SNR/10)*Pn1))
    #将噪声数据乘以k,
    random_values_we_need=random_values*k
    #计算新的噪声数据的功率
    Pn=np.sum(random_values_we_need**2)/len(random_values_we_need)
    #以下开始计算信噪比
    snr=10*math.log10(Ps/Pn)
    outdata = src + random_values_we_need
    # print("当前信噪比：",snr)
    return outdata
#%%
import librosa
Noise, sr_noise = librosa.load('/mnt/data/CCM/snndatabase/noise92/babble.wav', sr=19880)
def add_noise(sound,snr,noise=Noise):
    min_length = min(len(sound), len(noise))
    sound = sound[:min_length]
    noise = noise[:min_length]
    signal_energy = np.sum(sound ** 2)
    noise_energy = np.sum(noise ** 2)

    # 计算混合比例以达到特定SNR
    target_snr_db = snr  # 目标SNR为10dB
    target_snr_linear = 10 ** (target_snr_db / 10)  # 将dB转换为线性比例
    mixing_ratio = np.sqrt(signal_energy / (target_snr_linear * noise_energy))

    # 将声音和噪声按照混合比例混合
    mixed_audio = sound + mixing_ratio * noise
    return mixed_audio
#%%
segments = VAD(waveform, 8000)

if len(segments) != 0:
    waveform = waveform[segments[0][0]:segments[0][1]]
waveform = normalize_audio_peak(waveform, 1)
plt.figure()
plt.plot(waveform)
plt.show()
#%%
f=Freq
Ta=3.64*(f/1000)**(-0.8)-6.5*np.exp(-0.6*((f/1000)-3.3)**2)+0.001*(f/1000)**4;
# Ta[Ta < 0] = 0
#%%
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

#%%
after_wave=pre_emphasis(waveform)
plt.figure()
plt.plot(waveform)
plt.show()

plt.figure()
plt.plot(after_wave)
plt.show()

plt.figure()
waveform_1=add_noise(waveform,0)
after_wave_1=pre_emphasis(waveform_1)
plt.plot(after_wave_1)
plt.show()

segments = VAD(after_wave, 8000)
if len(segments) != 0:
    waveform_yuan = after_wave[segments[0][0]:segments[-1][1]]
plt.figure()
plt.plot(waveform_yuan)
plt.show()

segments = VAD(after_wave_1, 8000)
if len(segments) != 0:
    waveform_noise = after_wave_1[segments[0][0]:segments[-1][1]]
plt.figure()
plt.plot(waveform_noise)
plt.show()


#%%
sample_rate=8000
import librosa
import math
# cqtpec = cqt(waveform, sr=sample_rate,fmin=32,n_bins=83,hop_length=256)

# waveform_1=after_wave
waveform_1=after_wave_1
# waveform_1=add_noise(waveform,99)
# waveform_1=add_noise(after_wave,20)
# waveform_1=pre_emphasis(waveform_1)
segments = VAD(waveform_1, 8000)

if len(segments) != 0:
    waveform_1 = waveform_1[segments[0][0]:segments[-1][1]]
# waveform_1 = normalize_audio_peak(waveform_1, 1)
# waveform = normalize_audio_peak(waveform, 1)
# cqtpec = cqt(waveform, sr=sample_rate,fmin=32,n_bins=83,hop_length=96)
cqtpec = cqt(waveform_1, sr=sample_rate,fmin=32,n_bins=83,hop_length=128)
# cqtpec = cqt(mixed_audio, sr=sample_rate,fmin=32,n_bins=83,hop_length=96)
# popop=np.abs(cqtpec)
# Engry_1=(np.abs(cqtpec))**2
# Engry = librosa.amplitude_to_db(np.abs(cqtpec)**2)
# Engry = librosa.amplitude_to_db(np.abs(cqtpec), ref=np.max)

cqtm, phase = librosa.core.magphase(cqtpec)
# Engry = librosa.amplitude_to_db(cqtm,ref=np.max)

# Engry = abs(cqtpec)
Engry=cqtm
# Engry=np.log10(Engry_1)
# Engry=np.abs(cqtpec**2)
# Engry=librosa.amplitude_to_db(cqtpec, ref=np.max)

# cqtm, phase = librosa.core.magphase(cqtpec)
# Engry = librosa.amplitude_to_db(cqtm, ref=np.max)
# Engry = librosa.power_to_db(np.abs(cqtpec)**2)
# Engry=np.abs(cqtpec)
# Engry=10*np.log10(np.abs(cqtpec))
# Engry = librosa.amplitude_to_db(np.abs(cqtpec), ref=np.max)
# Engry=-Engry**2
Min_data=np.min(Engry)
Max_data=np.max(Engry)
# Engry=Engry-Min_data
# Engry=Engry-Min_data
# for idx in range(Engry.shape[0]):
#     Engry[idx,:][Engry[idx,:]<Ta[idx]]=0
# Min_data=np.min(Engry)
# Max_data=np.max(Engry)
data_per_file=(Engry-Min_data)/(Max_data-Min_data)
# data_per_file=Engry
cqtm, phase = librosa.core.magphase(cqtpec)
mag_cqt = librosa.amplitude_to_db(cqtm,ref=np.max)

# 绘制CQT频谱图
plt.figure(figsize=(10, 4))
librosa.display.specshow(mag_cqt, sr=8000, x_axis='time', y_axis='cqt_note')
plt.colorbar(label='Amplitude (dB)')
plt.title('CQT spectrogram')
plt.tight_layout()
plt.show()
#%%
Max_num=83
zeronum=abs(Max_num-data_per_file.shape[1]);
temp=np.pad(data_per_file,((0,0),(0,zeronum)))

#%%
data_per_file=temp

#%%
for idx in range(data_per_file.shape[0]):
    data_per_file[idx,:][data_per_file[idx,:]-Ta[idx]/100<0]=0
#%% 参数  平均频率
freq=[]
for idx in range(7):
    # temp=np.arange(2**(5+idx),2**(6+idx),((2**(6+idx)-2**(5+idx))/13));
    temp=np.linspace(2**(5+idx),2**(6+idx),13);
    freq.append(temp)
    
freq=np.stack(freq, axis=0)
freq=freq.reshape(-1)
freq=np.unique(freq)
freq=freq[0:84]
Freq=freq[0:data_per_file.shape[0]]
T=np.arange(0,data_per_file.shape[1],1)
plt.figure()
plt.pcolormesh(T, Freq, data_per_file, shading='gouraud')

plt.xlabel('Time Steps')

plt.ylabel('Freq bins')
plt.yticks(Freq)
plt.show()
#%%
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
#%%
import numpy as np
file='/mnt/data/CCM/snndatabase/CQT_a_list_real.txt'
a_list = np.zeros((83, ))
f = open(file, 'r')
content = f.readlines()
f.close()
#%%
# a_list=np.array(content)
row=0
for items in content:
    data_i = items.split()
    for x in data_i:
        a_list[row] = x
        row+=1
#%%
T=10
F_data=[]
T_data=[]
for time in range(data_per_file.shape[1]):
    T_temp=[]
    for frq in range(data_per_file.shape[0]):
        Timepoint,V_Statue=lzhikevich_model(T,data_per_file[frq][time]*100,a_list[frq],0.25,-65,8)
        # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 2)
        # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[1],0.25,-65,8)

      
        spike_array = np.zeros(T, dtype=bool)
        for i in Timepoint:
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
for time in range(test.shape[0]):
    temp= np.where(test[time,:,:] == True)
    nern=temp[0]
    timestep=temp[1]+T*time
    f.append(nern)
    t.append(timestep)
Freqq=np.concatenate(f,axis=0)
Timee=np.concatenate(t,axis=0)

plt.figure()
plt.plot(Timee,Freq[Freqq],'|', color='k')
plt.yticks(Freq)
T=np.arange(0,data_per_file.shape[1],10)
plt.xlim([0,np.arange(0,data_per_file.shape[1],1).shape[0]*10])
plt.show()
#%%
result_vertical = np.concatenate((Timee, Freqq), axis=0)
#%%
from spikingjelly.activation_based import encoding
encoder = encoding.PoissonEncoder()
T=10
TEST=[]

x = torch.from_numpy(data_per_file)
w, h = x.shape
out_spike = torch.full((T, w, h), 0, dtype=torch.bool)
for t in range(T):
    out_spike[t] = encoder(x)
TEST.append(np.array(out_spike))
test=np.stack(TEST, axis=0)
temp = torch.from_numpy(test).type(torch.FloatTensor)
test=temp.permute(0,3,2,1)
# test=temp.permute(0,2,3,1)
test=np.array(test)
test = np.squeeze(test)

#%%
import librosa
target_sr=8000
noise, sr_noise = librosa.load('/mnt/data/CCM/snndatabase/noise92/white.wav')
sound=waveform
min_length = min(len(sound), len(noise))
sound = sound[:min_length]
noise = noise[:min_length]
signal_energy = np.sum(sound ** 2)
noise_energy = np.sum(noise ** 2)

# 计算混合比例以达到特定SNR
target_snr_db = 0 # 目标SNR为10dB
target_snr_linear = 10 ** (target_snr_db / 10)  # 将dB转换为线性比例
mixing_ratio = np.sqrt(signal_energy / (target_snr_linear * noise_energy))

# 将声音和噪声按照混合比例混合
mixed_audio = sound + mixing_ratio * noise
#%%
import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

lif = neuron.LIFNode(tau=8.)
lif.reset()
x = torch.rand(size=[32]) * 4
T = 50
s_list = []
v_list = []
for t in range(T):
    s_list.append(lif(x).unsqueeze(0))
    v_list.append(lif.v.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)

visualizing.plot_2d_heatmap(array=np.asarray(v_list), title='Membrane Potentials', xlabel='Simulating Step',
                            ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
visualizing.plot_1d_spikes(spikes=np.asarray(s_list), title='Membrane Potentials', xlabel='Simulating Step',
                           ylabel='Neuron Index', dpi=200)
plt.show()

#%%
lif = neuron.LIFNode(tau=200.)
lif.reset()
x = torch.as_tensor(100.0)
T = 10
s_list = []
v_list = []
for t in range(T):
    s_list.append(lif(x))
    v_list.append(lif.v)

visualizing.plot_one_neuron_v_s(np.asarray(v_list), np.asarray(s_list), v_threshold=lif.v_threshold, v_reset=lif.v_reset,
                                dpi=200)
plt.show()
#%%
spike_train=np.asarray(s_list)
A = np.array(spike_train, dtype= bool)

#%% LIF
lif = neuron.LIFNode(tau=200.)
lif.reset()
T = 10
F_data = []
T_data = []
for time in range(data_per_file.shape[1]):
    T_temp = []
    for frq in range(data_per_file.shape[0]):
        s_list = []
        lif.reset()
        for t in range(T):
            s_list.append(lif(torch.as_tensor(data_per_file[frq][time] * 100)))
        spike_train = np.asarray(s_list)
        spike_array = np.array(spike_train, dtype=bool)
        T_temp = F_data.append(spike_array)
        # T_temp=np.stack(F_data,axis=0)
        pass
    T_temp = np.stack(F_data, axis=0)
    # T_temp=np.array(T_temp)
    T_data.append(T_temp)
    F_data = []
    T_temp = []

test = np.stack(T_data, axis=0)

#%%
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
#%%
import numpy as np
import math
import matplotlib.pyplot as plt
# def LIF_Model(T,I,tau):
T=10
tau=200
I=100
Timepoint=[]
u=0
theta=1
V_Statue=[]
v_reset=0
for idx in range(T):
    # u=u*math.pow(math.e,-1/tau)+I
    # u = u - (u - theta) / tau + I

    # u = u - (u - v_reset) / tau + I
    # spike = int(u >= theta)
    # u = v_reset * spike + (1. - spike) * u
    u = u + (I - u) / tau
    # u = u * (1. - 1. / tau) + I
    print(u)
    spike = int(u >= theta)
    # spike=np.array(u >= theta, dtype=int)
    u = v_reset * spike + (1. - spike) * u

    Timepoint.append(spike)
    V_Statue.append(u)
        # u = u * (1. - 1. / tau) + I
        # if u>theta:
        #     u=0
        #     Timepoint.append(idx)
        #
        # V_Statue.append(u)
        # if u>theta:
        #     u=0
        #     Timepoint.append(idx)
    # return np.array(Timepoint),np.array(V_Statue)


# Timepoint,V_Statue=LIF_Model(10,100,200.)
plt.figure()
plt.plot(V_Statue)
plt.show()

#%%
def LIF_Model(T,I,tau):

    Timepoint = []
    u = 0
    theta = 1
    V_Statue = []
    v_reset = 0
    for idx in range(T):
        u = u + (I - u) / tau
        spike = int(u >= theta)
        # u = v_reset * spike + (1. - spike) * u
        u = u - spike * theta
        Timepoint.append(spike)
        V_Statue.append(u)

    return Timepoint,V_Statue

Timepoint,V_Statue=LIF_Model(20,100,200)
plt.figure()
plt.plot(V_Statue)
plt.show()
#%%
def LIF_Model_new(T,I,tau):

    Timepoint = []
    u = 0
    theta = 1
    V_Statue = []
    v_reset = 0
    for idx in range(T):
        u+=(I-u)/tau
        spike = int(u >= theta)
        # u = v_reset * spike + (1. - spike) * u
        u = u - spike * theta
        Timepoint.append(spike)
        V_Statue.append(u)

    return Timepoint,V_Statue

Timepoint,V_Statue=LIF_Model_new(20,100,200)
plt.figure()
plt.plot(V_Statue)
plt.show()

#%%
T = 10
F_data = []
T_data = []
for time in range(data_per_file.shape[1]):
    T_temp = []
    for frq in range(data_per_file.shape[0]):
        Timepoint,V_Statue = LIF_Model(T, data_per_file[frq][time] * 100, 200)
        # Timepoint, time, membrane_potential = lif_neuron.simulate(input_current=data_per_file[frq][time] * 100, time=T)
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

test = np.stack(T_data, axis=0)
#%%
import numpy as np
import matplotlib.pyplot as plt

def LIF_Model(T, X, tau):
    H = np.zeros(T)  # 初始化膜电位数组
    V = np.zeros(T)  # 初始化动作电位数组
    Timepoint=[]
    v_reset=0
    for t in range(1, T):
        H[t] = H[t - 1] + (1 / tau) * (X - (V[t - 1] - v_reset))
        if H[t] >= 1.0:  # 判断是否达到阈值
            V[t] = 1.0  # 达到阈值，发放动作电位
            H[t] = v_reset  # 重置膜电位
            Timepoint.append(t)
        else:
            V[t] = 0.0  # 未达到阈值，不发放动作电位


    return Timepoint,V
#%%
# 模拟输入电流
T = 10  # 模拟时间步数
X = np.random.uniform(0, 1, T)  # 随机生成输入电流

# 模型参数
tau = 20  # 膜时间常数
v_reset = 0  # 重置电位

# 模拟LIF模型
Timepoint,spikes = LIF_Model(T, 100, 200, v_reset)

# 绘制模拟结果
plt.figure(figsize=(10, 5))
plt.plot(spikes, marker='o', linestyle='-', color='b')
plt.xlabel('Time Steps')
plt.ylabel('Spike')
plt.title('LIF Model Simulation')
plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt


class LIFNeuron:
    def __init__(self, membrane_resistance=10, membrane_time_scale=8, firing_threshold=1):
        self.membrane_resistance = membrane_resistance  # 膜电阻 (MΩ)
        self.membrane_time_scale = membrane_time_scale  # 膜时间常数 (ms)
        self.firing_threshold = firing_threshold  # 动作电位阈值 (mV)
        self.membrane_potential = 0  # 初始化膜电位

    def simulate(self, input_current, time):
        """
        模拟 LIF 模型
        :param input_current: 输入电流 (mA)
        :param time: 总的模拟时间 (ms)
        :return: 时间和膜电位的历史记录
        """
        dt = 1  # 时间步长 (ms)
        time_steps = int(time / dt)  # 总的时间步数
        membrane_potential_trace = []  # 膜电位的历史记录
        Timepoint=[]

        # 模拟每个时间步
        for idx in range(time_steps):
            # 更新膜电位
            dV = dt * (-(self.membrane_potential) + self.membrane_resistance * input_current) / self.membrane_time_scale
            self.membrane_potential += dV

            # 发放动作电位
            if self.membrane_potential >= self.firing_threshold:
                membrane_potential_trace.append(self.firing_threshold)  # 记录阈值
                self.membrane_potential = 0  # 重置膜电位
                Timepoint.append(idx)
            else:
                membrane_potential_trace.append(self.membrane_potential)  # 记录膜电位

        # return np.array(Timepoint),np.arange(0, time, dt), membrane_potential_trace
        return np.array(Timepoint), membrane_potential_trace


# 设置 LIF 神经元参数
lif_neuron = LIFNeuron(membrane_resistance=1, membrane_time_scale=200)

# 模拟 100ms 时间内，输入电流为 1.5mA 的情况
Timepoint, membrane_potential = lif_neuron.simulate(input_current=0.011742742935894057*100, time=10)

# 绘制膜电位随时间的变化
# plt.plot(time, membrane_potential)
plt.plot(membrane_potential)
plt.title('LIF Neuron Simulation')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.show()

#%%
T = 10
F_data = []
T_data = []
for time in range(data_per_file.shape[1]):
    T_temp = []
    for frq in range(data_per_file.shape[0]):
        # Timepoint,V_Statue = LIF_Model(T, data_per_file[frq][time] * 100, 200,0)
        inputdata=data_per_file[frq][time] * 100
        Timepoint,membrane_potential = lif_neuron.simulate(input_current=inputdata, time=T)
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

test = np.stack(T_data, axis=0)

#%%

import numpy as np

# Given parameters
sr = 8000  # Sample rate
fmin = 32  # Minimum frequency
n_bins = 84  # Number of bins
hop_length = 96  # Hop length

# Calculating the bins per octave (B)
# Using the relation: n_bins = B * log2(sr/(2*fmin))
B = n_bins / np.log2(sr/(2*fmin))


# Function to calculate center frequencies and bandwidths
def calculate_cqt_params(fmin, B, n_bins):
    center_frequencies = [fmin * (2 ** (k / B)) for k in range(n_bins)]
    bandwidths = [center_frequencies[i+1] - center_frequencies[i] for i in range(n_bins-1)]
    return center_frequencies, bandwidths

center_frequencies, bandwidths = calculate_cqt_params(fmin, B, n_bins)
# center_frequencies[:5], bandwidths[:5]  # Displaying first 5 for brevity
center_frequencies=np.array(center_frequencies)

#%% 导入数据

import numpy as np

import matplotlib.pyplot as plt

from librosa import cqt

import torch
import torchaudio
from scipy import signal
from spikingjelly.activation_based import encoding
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from scipy.signal import butter, lfilter

# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/0/109.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/1/1013.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/2/1015.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/3/1017.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/4/1019.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/5/141.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/6/1200.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/7/1510.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/8/16.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/9/128.wav'
# filename = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/10/2087.wav'

# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/0_bells5/002.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/1_bottle1/021.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/2_buzzer/028.wav'
filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/3_cymbals/023.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/4_horn/072.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/5_kara/090.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/6_metal15/082.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/7_phone4/044.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/8_ring/094.wav'
# filename = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/9_whistle1/030.wav'


NO=3
waveform, sample_rate = torchaudio.load(filename)
waveform=np.array(waveform)
waveform=waveform.reshape(-1)
savepath='/home/handsome/PythonProject/SNN/snnmodel/figure_draw/RWCP_10dB/CQT_encoding_xiu/'

if os.path.exists(savepath):
    print('ready exist')
else:
    print('ok I make it')
    os.makedirs(savepath)


Noise, sr_noise = librosa.load('/mnt/data/CCM/snndatabase/noise92/babble.wav', sr=19980)
def add_noise(sound,snr,noise=Noise):
    old_sample_rate = 19980
    SAMPLE_RATE = 8000
    noise = librosa.resample(noise, orig_sr=old_sample_rate, target_sr=SAMPLE_RATE)
    min_length = min(len(sound), len(noise))
    start_index = np.random.randint(0, len(noise) - len(sound) + 1)
    sound = sound[:min_length]
    # duan=len(noise)/min_length
    # rand_duan=np.random.randint(duan)
    noise = noise[start_index:start_index + len(sound)]
    # noise = noise[:min_length]
    # noise = noise[rand_duan*min_length:(rand_duan+1)*min_length]
    signal_energy = np.sum(sound ** 2)
    noise_energy = np.sum(noise ** 2)

    # 计算混合比例以达到特定SNR
    target_snr_db = snr  # 目标SNR为10dB
    target_snr_linear = 10 ** (target_snr_db / 10)  # 将dB转换为线性比例
    mixing_ratio = np.sqrt(signal_energy / (target_snr_linear * noise_energy))

    # 将声音和噪声按照混合比例混合
    mixed_audio = sound + mixing_ratio * noise
    return mixed_audio



waveform=add_noise(waveform,10)


plt.figure()
plt.plot(waveform)
# plt.savefig(savepath+'sound'+'.svg')
plt.show()
#% 计算
SAVAWAY='/home/handsome/PythonProject/SNN/snnmodel/figure_draw/RWCP_10dB/'
# 预处理
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

sounddata=waveform
sounddata=pre_emphasis(sounddata)

segments = VAD(sounddata, 8000)

if len(segments) != 0:
    sounddata = sounddata[segments[0][0]:segments[-1][1]]

sounddata = normalize_audio_peak(sounddata, 1)
plt.figure()
plt.plot(sounddata)
# plt.savefig(savepath+'pre_sound'+'.svg')
plt.show()

#% 提取特征
import librosa
from librosa import cqt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
cqtpec = cqt(sounddata, sr=sample_rate, fmin=32, n_bins=83, hop_length=96)
# Engry = abs(cqtpec)**2
cqtm, phase = librosa.core.magphase(cqtpec)
Engry = cqtm
# Engry = librosa.amplitude_to_db(cqtm, ref=np.max)

Min_data = np.min(Engry)
Max_data = np.max(Engry)
data_per_file = (Engry - Min_data) / (Max_data - Min_data)

sr = 8000  # Sample rate
fmin = 32  # Minimum frequency
n_bins = 83  # Number of bins
hop_length = 96  # Hop length

# Calculating the bins per octave (B)
# Using the relation: n_bins = B * log2(sr/(2*fmin))
B = n_bins / np.log2(sr/(2*fmin))


# Function to calculate center frequencies and bandwidths
def calculate_cqt_params(fmin, B, n_bins):
    center_frequencies = [fmin * (2 ** (k / B)) for k in range(n_bins)]
    bandwidths = [center_frequencies[i+1] - center_frequencies[i] for i in range(n_bins-1)]
    return center_frequencies, bandwidths

center_frequencies, bandwidths = calculate_cqt_params(fmin, B, n_bins)
# center_frequencies[:5], bandwidths[:5]  # Displaying first 5 for brevity
center_frequencies=np.array(center_frequencies)
Freq=center_frequencies
T = np.arange(0, data_per_file.shape[1]/8000, 1/8000)
# plt.figure()
ax = plt.figure().gca()
ax.pcolormesh(T*100, Freq, data_per_file, shading='gouraud')
# plt.yticks(Freq)
# plt.ylim([0, 4000])
ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.xlabel('Time (s)')

plt.ylabel('Frequency (Hz)')
# plt.savefig(savepath+'CQT'+'.svg')
plt.savefig(savepath+'Freq_feature_123'+str(NO)+'.pdf')
plt.show()
#%%
#% CQT_encoding
savepath=SAVAWAY+'CQT_encoding_xiu/'
if os.path.exists(savepath):
    print('ready exist')
else:
    print('ok I make it')
    os.makedirs(savepath)
T_list=[2,3,5,8,10]
for T in T_list:
    def lzhikevich_model(T, I, a, b, c, d):
        V_Statue = []
        v = -65  # 初始化膜电位
        u = b * v

        Timepoint = []

        spikes = 0
        for idx in range(T):  # 模拟1秒钟（1000毫秒）
            # v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + current)  # 使用Euler方法更新v
            v += (0.04 * v ** 2 + 5 * v + 140 - u + I)
            u += a * (b * v - u)  # 使用Euler方法更新u

            if v >= 30:  # 如果膜电位超过阈值，则发放脉冲
                v = c  # 重置膜电位
                u += d  # 更新恢复变量
                Timepoint.append(idx)
                V_Statue.append(30)
            else:

                V_Statue.append(v)
        # plt.figure()
        # plt.plot(np.array(V_Statue))
        # plt.show()
        return np.array(Timepoint), np.array(V_Statue)

    import numpy as np

    file = '/mnt/data/CCM/snndatabase/CQT_a_list_real.txt'
    a_list = np.zeros((83,))
    f = open(file, 'r')
    content = f.readlines()
    f.close()
    row=0
    for items in content:
        data_i = items.split()
        for x in data_i:
            a_list[row] = x
            row+=1


    F_data = []
    T_data = []
    for time in range(data_per_file.shape[1]):
        T_temp = []
        for frq in range(data_per_file.shape[0]):
            Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[frq], 0.25, -65, 8)
            # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 2)
            # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[1],0.25,-65,8)

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

    test = np.stack(T_data, axis=0)
    # #%%
    # Timepoint, V_Statue = lzhikevich_model(100,100, a_list[0], 0.25, -65, 8)
    # plt.figure()
    # plt.plot(V_Statue)
    # plt.show()

    f = []
    t = []
    for time in range(test.shape[0]):
        temp = np.where(test[time, :, :] == True)
        nern = temp[0]
        timestep = temp[1] + T * time
        f.append(nern)
        t.append(timestep)
    Freqq = np.concatenate(f, axis=0)
    Timee = np.concatenate(t, axis=0)

    ax = plt.figure().gca()
    plt.plot(Timee, Freq[Freqq], '|', color='k')
    # plt.yticks(Freq)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))


    plt.xlim([0, np.arange(0, data_per_file.shape[1], 1).shape[0] * T])
    plt.ylim([0, Freq[-1]])

    orgin=np.arange(0,np.arange(0, data_per_file.shape[1], 1).shape[0] * T,T).tolist()
    old=np.arange(0,np.arange(0, data_per_file.shape[1], 1).shape[0] *T/1000,T/1000).tolist()

    plt.xticks(orgin,old)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(savepath+'CQT_coding_'+str(NO)+'_T'+str(T)+'.svg')
    plt.show()
#%%
#% Freq_encoding
savepath=SAVAWAY+'Freq_encoding/'
if os.path.exists(savepath):
    print('ready exist')
else:
    print('ok I make it')
    os.makedirs(savepath)
from spikingjelly.activation_based import encoding
T_list=[2,3,5,8,10]
for T in T_list:
    encoder = encoding.PoissonEncoder()
    TEST=[]

    x = torch.from_numpy(data_per_file)
    w, h = x.shape
    out_spike = torch.full((T, w, h), 0, dtype=torch.bool)
    for t in range(T):
        out_spike[t] = encoder(x)
    TEST.append(np.array(out_spike))
    test=np.stack(TEST, axis=0)
    temp = torch.from_numpy(test).type(torch.FloatTensor)
    test=temp.permute(0,3,2,1)
    # test=temp.permute(0,2,3,1)
    test=np.array(test)
    test = np.squeeze(test)

    f = []
    t = []
    for time in range(test.shape[0]):
        temp = np.where(test[time, :, :] == True)
        nern = temp[0]
        timestep = temp[1] + T * time
        f.append(nern)
        t.append(timestep)
    Freqq = np.concatenate(f, axis=0)
    Timee = np.concatenate(t, axis=0)

    ax = plt.figure().gca()
    plt.plot(Timee, Freq[Freqq], '|', color='k')
    # plt.yticks(Freq)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))


    plt.xlim([0, np.arange(0, data_per_file.shape[1], 1).shape[0] * T])
    plt.ylim([0, Freq[-1]])

    orgin=np.arange(0,np.arange(0, data_per_file.shape[1], 1).shape[0] * T,T).tolist()
    old=np.arange(0,np.arange(0, data_per_file.shape[1], 1).shape[0] *T/1000,T/1000).tolist()

    plt.xticks(orgin,old)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(savepath+'Freq_coding_'+str(NO)+'_T'+str(T)+'.svg')
    plt.show()
#% LIF_encoding
savepath=SAVAWAY+'/LIF_encoding/'
if os.path.exists(savepath):
    print('ready exist')
else:
    print('ok I make it')
    os.makedirs(savepath)
class LIFNeuron:
    def __init__(self, membrane_resistance=10, membrane_time_scale=8, firing_threshold=1):
        self.membrane_resistance = membrane_resistance  # 膜电阻 (MΩ)
        self.membrane_time_scale = membrane_time_scale  # 膜时间常数 (ms)
        self.firing_threshold = firing_threshold  # 动作电位阈值 (mV)
        self.membrane_potential = 0  # 初始化膜电位

    def simulate(self, input_current, time):
        """
        模拟 LIF 模型
        :param input_current: 输入电流 (mA)
        :param time: 总的模拟时间 (ms)
        :return: 时间和膜电位的历史记录
        """
        dt = 1  # 时间步长 (ms)
        time_steps = int(time / dt)  # 总的时间步数
        membrane_potential_trace = []  # 膜电位的历史记录
        Timepoint=[]

        # 模拟每个时间步
        for idx in range(time_steps):
            # 更新膜电位
            dV = dt * (-(self.membrane_potential) + self.membrane_resistance * input_current) / self.membrane_time_scale
            self.membrane_potential += dV

            # 发放动作电位
            if self.membrane_potential >= self.firing_threshold:
                membrane_potential_trace.append(self.firing_threshold)  # 记录阈值
                self.membrane_potential = 0  # 重置膜电位
                Timepoint.append(idx)
            else:
                membrane_potential_trace.append(self.membrane_potential)  # 记录膜电位

        # return np.array(Timepoint),np.arange(0, time, dt), membrane_potential_trace
        return np.array(Timepoint), membrane_potential_trace


# 设置 LIF 神经元参数
lif_neuron = LIFNeuron(membrane_resistance=1, membrane_time_scale=200)

T_list=[2,3,5,8,10]
for T in T_list:
    F_data = []
    T_data = []
    for time in range(data_per_file.shape[1]):
        T_temp = []
        for frq in range(data_per_file.shape[0]):
            # Timepoint,V_Statue = LIF_Model(T, data_per_file[frq][time] * 100, 200,0)
            inputdata=data_per_file[frq][time] * 100
            Timepoint,membrane_potential = lif_neuron.simulate(input_current=inputdata, time=T)
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

    test = np.stack(T_data, axis=0)

    f = []
    t = []
    for time in range(test.shape[0]):
        temp = np.where(test[time, :, :] == True)
        nern = temp[0]
        timestep = temp[1] + T * time
        f.append(nern)
        t.append(timestep)
    Freqq = np.concatenate(f, axis=0)
    Timee = np.concatenate(t, axis=0)

    ax = plt.figure().gca()
    plt.plot(Timee, Freq[Freqq], '|', color='k')
    # plt.yticks(Freq)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.xlim([0, np.arange(0, data_per_file.shape[1], 1).shape[0] * T])
    plt.ylim([0, Freq[-1]])

    orgin = np.arange(0, np.arange(0, data_per_file.shape[1], 1).shape[0] * T, T).tolist()
    old = np.arange(0, np.arange(0, data_per_file.shape[1], 1).shape[0] * T / 1000, T / 1000).tolist()

    plt.xticks(orgin, old)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(savepath + 'Freq_coding_' + str(NO) + '_T' + str(T) + '.svg')
    plt.show()
#% Lzh _encoding
savepath=SAVAWAY+'Lzh_encoding/'
if os.path.exists(savepath):
    print('ready exist')
else:
    print('ok I make it')
    os.makedirs(savepath)
T_list=[2,3,5,8,10]
for T in T_list:
    def lzhikevich_model(T, I, a, b, c, d):
        V_Statue = []
        v = -65  # 初始化膜电位
        u = b * v

        Timepoint = []

        spikes = 0
        for idx in range(T):  # 模拟1秒钟（1000毫秒）
            # v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + current)  # 使用Euler方法更新v
            v += (0.04 * v ** 2 + 5 * v + 140 - u + I)
            u += a * (b * v - u)  # 使用Euler方法更新u

            if v >= 30:  # 如果膜电位超过阈值，则发放脉冲
                v = c  # 重置膜电位
                u += d  # 更新恢复变量
                Timepoint.append(idx)
                V_Statue.append(30)
            else:

                V_Statue.append(v)
        # plt.figure()
        # plt.plot(np.array(V_Statue))
        # plt.show()
        return np.array(Timepoint), np.array(V_Statue)

    import numpy as np

    file = '/mnt/data/CCM/snndatabase/CQT_a_list_real.txt'
    a_list = np.zeros((83,))
    f = open(file, 'r')
    content = f.readlines()
    f.close()
    row=0
    for items in content:
        data_i = items.split()
        for x in data_i:
            a_list[row] = x
            row+=1

    F_data = []
    T_data = []
    for time in range(data_per_file.shape[1]):
        T_temp = []
        for frq in range(data_per_file.shape[0]):
            Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 2)
            # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 2)
            # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[1],0.25,-65,8)

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

    test = np.stack(T_data, axis=0)
    # #%%
    # Timepoint, V_Statue = lzhikevich_model(100,100, a_list[0], 0.25, -65, 8)
    # plt.figure()
    # plt.plot(V_Statue)
    # plt.show()

    f = []
    t = []
    for time in range(test.shape[0]):
        temp = np.where(test[time, :, :] == True)
        nern = temp[0]
        timestep = temp[1] + T * time
        f.append(nern)
        t.append(timestep)
    Freqq = np.concatenate(f, axis=0)
    Timee = np.concatenate(t, axis=0)

    ax = plt.figure().gca()
    plt.plot(Timee, Freq[Freqq], '|', color='k')
    # plt.yticks(Freq)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))


    plt.xlim([0, np.arange(0, data_per_file.shape[1], 1).shape[0] * T])
    plt.ylim([0, Freq[-1]])

    orgin=np.arange(0,np.arange(0, data_per_file.shape[1], 1).shape[0] * T,T).tolist()
    old=np.arange(0,np.arange(0, data_per_file.shape[1], 1).shape[0] *T/1000,T/1000).tolist()

    plt.xticks(orgin,old)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(savepath+'CQT_coding_'+str(NO)+'_T'+str(T)+'.svg')
    plt.show()
#%%