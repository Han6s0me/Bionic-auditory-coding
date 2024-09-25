from scipy import signal
import numpy as np

import torch
import torchaudio
import matplotlib.pyplot as plt

from spikingjelly.activation_based import encoding
from spikingjelly import visualizing
import os

from librosa import cqt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import librosa
from argparse import ArgumentParser
import csv
#from adabelief_pytorch import AdaBelief

parser = ArgumentParser()
parser.add_argument('--spec', type=str, default='cqt', help='spectrogram type')
parser.add_argument('--net', type=str, default='snn', help='network type')
parser.add_argument('--split', type=str, default='raw', help='data split type')
parser.add_argument('--encoding', type=str, default='BAN', help='data split type')


args = parser.parse_args()

# 参数
train_dir = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/'
# train_dir = '/mnt/data/CCM/snndatabase/RWCP_train_8k_real/'
# train_dir = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all_all/'
# if args.split != 'raw':
#     train_dir = './snndatabase/TID_DATA_8k_all_all'
test_dir = '/mnt/data/CCM/snndatabase/TID_TEST_8k_all/'
# test_dir='/mnt/data/CCM/snndatabase/RWCP_test_8k_real/'
# train_dir = '/mnt/data/CCM/snndatabase/RWCP_train_8k/'
Max_num = 83
T = 10
batch_size = 64
learning_rate = 1e-3
tau = 2.0
train_epoch = 200
classnumber = 11
savepath = '/home/handsome/PythonProject/SNN/snnmodel/Model_TID_R_MSE_abcd/Cochlea_Coding_Eng_10/TEST_for_BAN_Blist/TID_canshu20_37/'
# savepath = '/home/handsome/PythonProject/SNN/snnmodel/Model_TID_R_MSE_abcd/Cochlea_Coding_Eng_10/TEST_for_MODEL_RWCP_VAD0.05/LIF_real/'
# savepath='./snnmodel/Resultmodel_VAD_big/Cochlea_Coding_Eng/T20/'
# random_state=60
# class LabelSmoothingCrossEntropy(torch.nn.Module):
#     def __init__(self, smoothing=0.1):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.smoothing = smoothing
#
#     def forward(self, input, target):
#         log_probs = F.log_softmax(input, dim=-1)
#         target = F.one_hot(target, num_classes=input.size(-1))
#         target = target.float()
#         target = target * (1 - self.smoothing) + self.smoothing / input.size(-1)
#         return -(target * log_probs).sum(dim=-1).mean()
#
# criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
# criterion = torch.nn.CrossEntropyLoss()
import random


def generate_random_number():
    return random.choice([-1, 0, 5, 10, 20])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


# %
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



def preemphasis(y, coef=0.97, zi=None, return_zf=False):
    """Pre-emphasize an audio signal with a first-order auto-regressive filter:

        y[n] -> y[n] - coef * y[n-1]


    Parameters
    ----------
    y : np.ndarray
        Audio signal

    coef : positive number
        Pre-emphasis coefficient.  Typical values of ``coef`` are between 0 and 1.

        At the limit ``coef=0``, the signal is unchanged.

        At ``coef=1``, the result is the first-order difference of the signal.

        The default (0.97) matches the pre-emphasis filter used in the HTK
        implementation of MFCCs [#]_.

        .. [#] http://htk.eng.cam.ac.uk/

    zi : number
        Initial filter state.  When making successive calls to non-overlapping
        frames, this can be set to the ``zf`` returned from the previous call.
        (See example below.)

        By default ``zi`` is initialized as ``2*y[0] - y[1]``.

    return_zf : boolean
        If ``True``, return the final filter state.
        If ``False``, only return the pre-emphasized signal.

    Returns
    -------
    y_out : np.ndarray
        pre-emphasized signal

    zf : number
        if ``return_zf=True``, the final filter state is also returned
    """

    b = np.asarray([1.0, -coef], dtype=y.dtype)
    a = np.asarray([1.0], dtype=y.dtype)

    if zi is None:
        # Initialize the filter to implement linear extrapolation
        zi = 2 * y[..., 0] - y[..., 1]

    zi = np.atleast_1d(zi)

    y_out, z_f = scipy.signal.lfilter(b, a, y, zi=np.asarray(zi, dtype=y.dtype))

    if return_zf:
        return y_out, z_f

    return y_out



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
    # win = 0.05
    # step = 0.05
    # win = 0.025
    # step = 0.025

    # win = 0.02
    # step = 0.02 #gai

    # win = 0.01
    # step = 0.01

    # win = 0.1
    # step = 0.1

    # win = 0.2
    # step = 0.2
    # win = 0.2
    # step = 0.2
    win = 0.05
    step = 0.05

    Eor = ShortTimeEnergy(signal, int(win * fs), int(step * fs));
    Cor = SpectralCentroid(signal, int(win * fs), int(step * fs), fs);
    E = scipy.signal.medfilt(Eor[:, 0], 5)
    E = scipy.signal.medfilt(E, 5)
    C = scipy.signal.medfilt(Cor[:, 0], 5)
    C = scipy.signal.medfilt(C, 5)
    # E = scipy.signal.medfilt(Eor[:, 0], 3)
    # E = scipy.signal.medfilt(E, 3)
    # C = scipy.signal.medfilt(Cor[:, 0], 3)
    # C = scipy.signal.medfilt(C, 3)

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
    Hist = np.histogram(C, bins=10)  # bins=10 3
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3)  # 3 10
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


Noise, sr_noise = librosa.load('/mnt/data/CCM/snndatabase/noise92/babble.wav', sr=19980)


def add_noise(sound, snr, noise=Noise):
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
            # random_number = generate_random_number()
            random_number = -1
            if random_number == -1:
                sounddata = sounddata
            else:
                sounddata = add_noise(sounddata, random_number)
            # sounddata = pre_emphasis(sounddata)
            sounddata = preemphasis(sounddata,coef=0.95)

            segments = VAD(sounddata, 8000)
            if len(segments) != 0:
                # sounddata = sounddata[segments[0][0]:segments[-1][1]]
                differences = [max(sublist) - min(sublist) for sublist in segments]
                # 找到最大差值的索引
                max_diff_index = differences.index(max(differences))
                # 获取差值最大的子列表
                list_with_max_diff = segments[max_diff_index]
                sounddata = sounddata[list_with_max_diff[0]:list_with_max_diff[1]]
            sounddata = normalize_audio_peak(sounddata, 1)
            if 'cqt' in args.spec:
                print('Using constant-Q transform')
                cqtpec = cqt(sounddata, sr=sample_rate, fmin=32, n_bins=83, hop_length=96)
                # Engry = abs(cqtpec)**2
                # Engry=np.log10(np.abs(cqtpec))
                cqtm, phase = librosa.core.magphase(cqtpec)
                Engry = cqtm
                # Engry = (np.abs(cqtm))**2
                # Engry = librosa.amplitude_to_db(cqtm, ref=np.max)
            elif args.spec == 'mel':
                # print("Using mel spectrogram")
                mel_spectrogram = librosa.feature.melspectrogram(y=sounddata, sr=sample_rate, n_mels=83, hop_length=96,
                                                                 n_fft=192)

                # 转换为对数尺度
                Engry = mel_spectrogram
            elif args.spec == 'mel2':
                print("Using mel spectrogram")
                mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=192,
                    hop_length=96,
                    n_mels=83
                )

                # 应用变换
                Engry = mel_spectrogram(torch.from_numpy(sounddata)).numpy()
                # Engry = librosa.power_to_db(mel_spectrogram, ref=np.max)
            elif args.spec == 'stft':
                print("Using naive stft")
                # stft_result = librosa.stft(y=sounddata, n_fft=164, hop_length=32)
                stft_result = librosa.core.stft(y=sounddata, n_fft=164, hop_length=96)
                magnitude, phase = librosa.magphase(stft_result)
                # Engry = librosa.amplitude_to_db(magnitude, ref=np.max)
                Engry = magnitude

            Min_data = np.min(Engry)
            Max_data = np.max(Engry)
            data_per_file = (Engry - Min_data) / (Max_data - Min_data)
            # pe = encoding.PoissonEncoder()
            # data_per_file=Engry

            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
            # print(file_nums_count, '\n')
        folder_nums += 1
    # data = np.array(data)

    labels = np.array(labels)
    return data, labels


def data_pad(train_data):
    # #%%  计算能量
    # TEST_engry=train_data[0]
    # #%%
    # # TEST_engry_mean=TEST_engry.mean(axis=1)
    # # TEST=[]
    # cell_array = np.empty((len(train_data),), dtype=object)
    # for idx in range(len(train_data)):
    #     TEST_engry = train_data[idx]
    #     # TEST_engry_mean = TEST_engry.mean(axis=1)
    #     # TEST.append(TEST_engry_mean)
    #     # TEST.append(np.array(TEST_engry, dtype=np.object))
    #     cell_array[idx]=TEST_engry
    # # test=np.stack(TEST, axis=0)
    # # result_engry=test
    # # result_engry=np.array(TEST)
    # #%%
    # from scipy.io import savemat
    # # savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Bioinspired_cochlea_coding/Energy_TID.mat', {'train_data': result_engry})
    # # savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Bioinspired_cochlea_coding/label_Energy_TID.mat', {'train_labels': train_labels})
    # savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Bioinspired_cochlea_coding/Energy_TID_training.mat', {'my_cell_array': cell_array})
    # #%%
    # for idx in range(len(train_data)):
    #     temp=train_data[idx].shape[1];
    #     if temp>Max_num:
    #         Max_num=temp

    TEST = []
    for idx in range(len(train_data)):
        if Max_num > train_data[idx].shape[1]:
            zeronum = abs(Max_num - train_data[idx].shape[1]);
            temp = np.pad(train_data[idx], ((0, 0), (0, zeronum)))
            # TEST.append(temp)
        else:
            temp = train_data[idx][:, 0:Max_num]
        TEST.append(temp)

    # test=np.array(TEST)
    test = np.stack(TEST, axis=0)
    train_data = test
    return train_data


# %  lzhikevich简单写
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
        V_Statue.append(v)
    # plt.figure()
    # plt.plot(np.array(V_Statue))
    # plt.show()
    return np.array(Timepoint), np.array(V_Statue)


# % 只做了a的优化
# T = 10
def BAN_encoding(train_data, a_list):
    TEST = []
    for idx in range(train_data.shape[0]):
        data_per_file = train_data[idx, :, :]
        F_data = []
        T_data = []
        for time in range(data_per_file.shape[1]):
            T_temp = []
            for frq in range(data_per_file.shape[0]):
                # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[frq,0], a_list[frq,1], a_list[frq,2], a_list[frq,3])
                temp = int(data_per_file[frq][time] * 10)
                # if temp == 10:
                #     temp = 9
                # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[frq, temp],
                #                                        a_list[frq, temp + 10],
                #                                        a_list[frq, temp + 20], a_list[frq, temp + 30])
                temp = int(data_per_file[frq][time] * 20)
                if temp == 20:
                    temp = 19
                Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[frq, temp],
                                                       a_list[frq, temp + 20],
                                                       a_list[frq, temp + 40], a_list[frq, temp + 60])

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
        TEST.append(test)
        print(idx)

    test123 = np.stack(TEST, axis=0)
    train_data = test123
    return train_data
def BAN_OLD_encoding(train_data, a_list):
    TEST = []
    for idx in range(train_data.shape[0]):
        data_per_file = train_data[idx, :, :]
        F_data = []
        T_data = []
        for time in range(data_per_file.shape[1]):
            T_temp = []
            for frq in range(data_per_file.shape[0]):
                # Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[frq], 0.25, -65, 8)
                Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, a_list[frq], 0.25, -65, 8)
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
        TEST.append(test)
        print(idx)

    test123 = np.stack(TEST, axis=0)
    train_data = test123
    return train_data
def IZH_encoding(train_data):
    TEST = []
    for idx in range(train_data.shape[0]):
        data_per_file = train_data[idx, :, :]
        F_data = []
        T_data = []
        for time in range(data_per_file.shape[1]):
            T_temp = []
            for frq in range(data_per_file.shape[0]):
                Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 2) # 0.02 0.25 -65 8

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
        TEST.append(test)
        print(idx)

    test123 = np.stack(TEST, axis=0)
    train_data = test123
    return train_data
# def IZH_encoding(train_data):
#     TEST = []
#     for idx in range(train_data.shape[0]):
#         data_per_file = train_data[idx, :, :]
#         T_data = []  # List to store time data
#         for time in range(data_per_file.shape[1]):
#             F_data = []  # List to store frequency data for each time point
#             for frq in range(data_per_file.shape[0]):
#                 Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 8)
#
#                 spike_array = np.zeros(T, dtype=bool)
#                 for i in Timepoint:
#                     spike_array[i] = True
#                 F_data.append(spike_array)  # Properly append spike data to frequency data list
#
#             # Stack frequency data to form a time slice
#             T_slice = np.stack(F_data, axis=0)
#             T_data.append(T_slice)  # Append the time slice to time data list
#
#         # Stack all time slices to form complete data for one file
#         test = np.stack(T_data, axis=0)
#         TEST.append(test)
#         print(idx)
#
#     # Stack all test data to recreate the train_data structure
#     train_data = np.stack(TEST, axis=0)
#     return train_data

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
                Timepoint = []

                # 模拟每个时间步
                for idx in range(time_steps):
                    # 更新膜电位
                    dV = dt * (-(
                        self.membrane_potential) + self.membrane_resistance * input_current) / self.membrane_time_scale
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
def LIF_encoding(train_data):
    TEST = []
    lif_neuron = LIFNeuron(membrane_resistance=1, membrane_time_scale=200)
    for idx in range(train_data.shape[0]):
        data_per_file = train_data[idx, :, :]
        F_data = []
        T_data = []
        for time in range(data_per_file.shape[1]):
            T_temp = []
            for frq in range(data_per_file.shape[0]):
                # Timepoint, V_Statue = LIF_Model(T, data_per_file[frq][time] * 100, 200)
                Timepoint, V_Statue = lif_neuron.simulate(data_per_file[frq][time] * 100, time=T)
                spike_array = np.zeros(T, dtype=bool)
                for i in Timepoint:
                    spike_array[i] = True
                T_temp = F_data.append(spike_array)
                pass
            T_temp = np.stack(F_data, axis=0)
            # T_temp=np.array(T_temp)
            T_data.append(T_temp)
            F_data = []
            T_temp = []
        test = np.stack(T_data, axis=0)
        TEST.append(test)
        print(idx)

    test123 = np.stack(TEST, axis=0)
    train_data = test123
    return train_data
def POISSON_encoding(train_data):
    encoder = encoding.PoissonEncoder()
    TEST = []
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
    test123 = np.stack(TEST, axis=0)
    train_data = test123
    # train_data = train_data.permute(0, 3, 2, 1)
    temp = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_data = temp.permute(0, 3, 2, 1)
    train_data = np.array(train_data)
    # train_data = np.squeeze(train_data)

    return train_data


# train_dir=r'D:\浙大项目\SNN\Database\DATA\DATA_WAV'
# train_dir='/mnt/data/CCM/snndatabase/DATA_WAV/'
# train_dir = '/mnt/data/CCM/snndatabase/TID_DATA_8k/'
encoding_cache_path = os.path.join(savepath, args.spec+ '-' +args.encoding+ '-' +str(T)+'-encoding_cache.npy')
if os.path.exists(encoding_cache_path):
    print('Loading encoding cache from', encoding_cache_path)
    encoding_cache = np.load(encoding_cache_path, allow_pickle=True).item()
    train_data = encoding_cache['train_data']
    train_labels = encoding_cache['train_labels']
    real_test_data = encoding_cache['real_test_data']
    real_test_labels = encoding_cache['real_test_labels']
else:
    print('Generating encoding cache to', encoding_cache_path)
    train_data, train_labels = read_origindata(train_dir)
    real_test_data, real_test_labels = read_origindata(test_dir)
    train_data = data_pad(train_data)
    real_test_data = data_pad(real_test_data)
    # % 参数  平均频率
    # freq = []
    # for idx in range(7):
    #     # temp=np.arange(2**(5+idx),2**(6+idx),((2**(6+idx)-2**(5+idx))/13));
    #     temp = np.linspace(2 ** (5 + idx), 2 ** (6 + idx), 13);
    #     freq.append(temp)
    #
    # freq = np.stack(freq, axis=0)
    # freq = freq.reshape(-1)
    # freq = np.unique(freq)
    # freq = freq[0:84]
    # Freq = freq[0:train_data.shape[1]]

    import numpy as np
    if args.encoding=='BAN':
        # file = '/home/handsome/MatlabProject/CQT_a_b_c_d_list_real_fenduan_18.txt'
        # file = '/home/handsome/MatlabProject/CQT_a_b_c_d_list_real_fenduan_59.txt'
        file = '/home/handsome/MatlabProject/CQT_a_b_c_d_list_real_fenduan20_37.txt'
        # file='/mnt/data/CCM/snndatabase/CQT_a_list.txt'
        #
        # a_list = np.zeros((83, 40))
        a_list = np.zeros((83, 80))
        # a_list = np.zeros((84, 40))
        f = open(file, 'r')
        content = f.readlines()
        f.close()
        # a_list=np.array(content)+
        row = 0
        for items in content:
            data_i = items.split()
            print(row)
            idx = 0
            for x in data_i:
                a_list[row][idx] = x
                idx += 1
            row += 1

        train_data = BAN_encoding(train_data, a_list)
        real_test_data = BAN_encoding(real_test_data, a_list)
    elif args.encoding=='BAN_OLD':

        # file = '/mnt/data/CCM/snndatabase/CQT_a_list.txt'
        file = '/home/handsome/MatlabProject/CQT_a_list_real.txt'
        a_list = np.zeros((83,))
        f = open(file, 'r')
        content = f.readlines()
        f.close()

        # a_list=np.array(content)
        row = 0
        for items in content:
            data_i = items.split()
            for x in data_i:
                a_list[row] = x
                row += 1

        train_data = BAN_OLD_encoding(train_data, a_list)
        real_test_data = BAN_OLD_encoding(real_test_data, a_list)
    elif args.encoding =='LIF':

        train_data = LIF_encoding(train_data)
        real_test_data = LIF_encoding(real_test_data)
    elif args.encoding == 'IZH':
        train_data = IZH_encoding(train_data)
        real_test_data = IZH_encoding(real_test_data)
    elif args.encoding == 'POISSON':
        train_data = POISSON_encoding(train_data)
        real_test_data = POISSON_encoding(real_test_data)


    np.save(encoding_cache_path,
            {'train_data': train_data, 'train_labels': train_labels, 'real_test_data': real_test_data,
             'real_test_labels': real_test_labels})

# #%% 保存数据
# from scipy.io import savemat
# savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Bioinspired_cochlea_coding/train_data_10T_real.mat', {'train_data': train_data})
# savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Bioinspired_cochlea_coding/label_data_10T_real.mat', {'train_labels': train_labels})
# #%%
# #%%加载数据
# from scipy import signal
# import numpy as np
#
# import torch
# import torchaudio
# import matplotlib.pyplot as plt
#
# from spikingjelly.activation_based import encoding
# from spikingjelly import visualizing
# import os
#
# from librosa import cqt
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import librosa
#
# # 参数
# train_dir = '/mnt/data/CCM/snndatabase/TID_DATA_8k_all/'
# # train_dir = '/mnt/data/CCM/snndatabase/RWCP_train_8k/'
# T = 10
# batch_size = 64
# learning_rate = 1e-2
# tau = 4.0
# train_epoch = 200
# savepath='./snnmodel/Model_TID_R_MSE/Cochlea_Coding_Eng/T10/'
# Max_num=83
# from scipy.io import loadmat
#
# # 读取MATLAB文件
# mat_data = loadmat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Bioinspired_cochlea_coding/train_data_10T.mat')
# mat_data_label = loadmat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Bioinspired_cochlea_coding/label_data_10T.mat')
#
# # mat_data 是一个字典，其中包含MATLAB文件中的变量
# # 你可以通过变量名访问数据，例如：
# variable_name = 'train_data'
# variable_name_label = 'train_labels'
# train_data = mat_data[variable_name]
# train_labels = mat_data_label[variable_name_label]
# train_labels=np.squeeze(train_labels)

# % 数据划分
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
name_result=savepath + 'summary_'  + args.spec+ '-'+args.encoding+'-' +args.net+'-'+str(T) + '.csv'

with open(name_result, 'a') as file:
    summary = {
        'best_acc': 0,
        'best_loss': 0,
        'real_test_acc': 0,
        'mode_path': 0,
    }
    writer = csv.DictWriter(file, fieldnames=summary.keys())
    writer.writeheader()  # 写入头部，即字段名
# %
for id in range(10):
    random_state = (id + 1) * 10
    # (X_train, X_test, Y_train, Y_test) = train_test_split(train_data, train_labels, test_size=0.5,
    #                                                       random_state=random_state)


    # % 数据封装
    # batch_size = 64
    # %

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
        # print(x_train_data.size(), y_train_data.size(), x_test_data.size(), y_test_data.size())

        # 将数据tensor和标签tensor包装成Dataset类
        train_dataset = TensorDataset(x_train_data, y_train_data)
        test_dataset = TensorDataset(x_test_data, y_test_data)

        # 将dataset传入DataLoader中，shuffle使得每个epoch中的样本生成顺序不一样，也就是不依次取样本
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


    # % 构建网络

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron,surrogate, functional, layer
    import torch.nn.functional as F


    def smooth_labels(labels, smoothing=0.1):
        # 类别数
        num_classes = labels.shape[1]

        # 计算平滑标签
        smooth_value = smoothing / num_classes
        with torch.no_grad():
            smooth_labels = labels * (1 - smoothing) + smooth_value
        return smooth_labels


    class SNNforauditory(nn.Module):
        def __init__(self, tau):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(83, 64, kernel_size=9 ,stride=1, padding=4),  # 假设输入为单通道
                # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
                # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.sigmoid()),
                # nn.AvgPool1d(kernel_size=2),
                nn.MaxPool1d(kernel_size=3),
                # layer.Conv1d(83, 64, kernel_size=9 ,stride=1, padding=4),  # 假设输入为单通道
                # layer.MaxPool1d(kernel_size=1),
                neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
                # neuron.ParametricLIFNode(init_tau=tau),
                # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                # layer.Dropout(0.6)

            )
            # # 其中 channels = 1, height = 1 (一维信号转二维处理时的高度), width = signal_length
            # self.conv2 = nn.Sequential(
            #     nn.Conv2d(1, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),  # channels = 1, 使用二维卷积
            #     # nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(3, 3), padding=(3, 3)),  # channels = 1, 使用二维卷积
            #     nn.MaxPool2d(kernel_size=(2, 2)),  # 对宽度维度进行池化
            #     neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
            #     # layer.Dropout(0.6)
            # )
            # self.conv2 = nn.Sequential(
            #     nn.Conv1d(64, 32, kernel_size=9, stride=1, padding=4),
            #     nn.MaxPool1d(kernel_size=2),
            #     neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
            #     layer.Dropout(0.6)
            # )

            self.fc = nn.Sequential(
                nn.Flatten(),
                # layer.Flatten(),
                layer.Dropout(0.5),
                # nn.Linear(83 * Max_num, 1024, bias=True),
                nn.Linear(64 * (Max_num // 3), 1024, bias=False),
                # layer.Linear(64 *(Max_num // 2), 1024, bias=False),
                # nn.Linear(107584, 1024, bias=True),
                # nn.Linear(512 , 1024, bias=True),
                # nn.Linear(64 * (Max_num // 2) * (Max_num // 2), 1024, bias=True),  # 更新 Flatten 后的维度
                # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
                # neuron.ParametricLIFNode(init_tau=tau),
                # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.sigmoid()),

                layer.Dropout(0.5),
                #
                # layer.Dropout(0.7),
                # nn.Linear(2048, 2048, bias=True),
                # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
                # layer.Dropout(0.5),
                # nn.Linear(1024, 1024, bias=True),
                # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
                # layer.Dropout(0.5),

            )
            self.fc1 = nn.Sequential(
                # layer.Dropout(0.8),
                nn.Linear(1024, classnumber, bias=False),
                # layer.Linear(1024,11,bias=False),
                # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
                # neuron.ParametricLIFNode(init_tau=tau),
                # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.sigmoid()),

            )

        def forward(self, x: torch.Tensor):
            # print(f"Input shape: {x.shape}")
            x = self.conv1(x)
            # x = self.conv2(x)
            # print(f"After conv2 shape: {x.shape}")
            x = self.fc(x)
            # print(f"After fc shape: {x.shape}")
            x = self.fc1(x)
            # print(f"Output shape: {x.shape}")
            return x


    class RNNauditory(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(input_size=83 * Max_num, hidden_size=256, batch_first=False)
            self.fc = nn.Linear(256, 11, bias=True)

        def forward(self, x: torch.Tensor):
            outputs, (hn, cn) = self.rnn(x.flatten(2))
            out = self.fc(hn[0])
            out = torch.softmax(out, dim=-1)
            return out


    def eval_model(net, test_loader, T):
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
                test_label_one_hot = F.one_hot(test_label, classnumber).float()
                # test_label_one_hot=test_label
                # test_label_one_hot = F.one_hot(test_label.unsqueeze(0).to(torch.int64), 11).float()
                # test_data_seq = test_data.permute(2, 0, 1)
                # test_data_seq = test_data.permute(1,0,2,3)
                test_data_seq = test_data.permute(3, 0, 2, 1)
                # test_data_seq = test_data.permute(1, 0, 2, 3)

                # test_data_seq = test_data_seq.unsqueeze(2)

                test_output = 0
                # for t in range(T):
                #     test_output +=net(test_data_seq[t])  # batch_size x 4 tensor
                # test_output = test_output / T
                if args.net == 'snn':
                    for t in range(T):
                        # encoded_data=test_data[:,t,:,:]
                        # encoded_data = encoder(test_data)
                        # test_output += net(encoded_data)
                        test_output += net(test_data_seq[t])
                    test_output = test_output / T
                elif args.net == 'rnn':
                    test_output = net(test_data_seq)
                elif args.net == 'cnn':
                    test_output = net(test_data_seq)

                val_loss = F.mse_loss(test_output, test_label_one_hot)
                # test_label_one_hot = test_label_one_hot.squeeze(0)
                # test_label_one_hot=test_label
                # val_loss = F.cross_entropy(test_output, test_label_one_hot)
                # val_loss=criterion(test_output, test_label_one_hot)
                val_losses.append(val_loss.item())
                correct_sum += (test_output.max(1)[1] == test_label).float().sum().item()  # 预测正确的样本数
                # correct_sum += (test_output.argmax(dim=1) == test_label).float().sum().item()
                test_sum += test_label.numel()  # numel()用来返回数组中元素的个数
                # print('test sample numbers:', test_sum)
                functional.reset_net(net)

            epoch_val_loss = np.array(val_losses).mean()
            test_accuracy = correct_sum / test_sum

            print('Epoch %s' % (epoch + 1), 'test_loss:', '%.5f' % epoch_val_loss, 'test_accuracy:',
                  '%.5f' % test_accuracy)
            return test_accuracy, epoch_val_loss


    # %
    # train_loader, test_loader = data_package(X_train, Y_train, X_test, Y_test, batch_size)
    # real_test_loader = DataLoader(
    #     TensorDataset(torch.from_numpy(real_test_data).float(), torch.from_numpy(real_test_labels)),
    #     batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=True)
    real_test_loader = DataLoader(
        TensorDataset(torch.from_numpy(real_test_data).float(), torch.from_numpy(real_test_labels)),
        batch_size=batch_size, shuffle=True)
    test_loader = real_test_loader
    # % 网络训练   data的格式[N,t,F,T]---t是语音时间，F是频率，T是神经元仿真时间，N是样本

    device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    print('Total GPU numbers:' + str(torch.cuda.device_count()), '\n', 'Being uesd GPU:' + str(device))
    torch.cuda.manual_seed(0)
    # batch_size = 64
    # 学习率
    # learning_rate = 1e-2

    # 仿真时长，T 越大，仿真占用显存越大
    # T = 15
    # LIF神经元的时间常数tau，tau 越大，拟合速度越慢
    # tau = 4.0
    # 训练轮数，经实验，一般需要设置为 10*tau 以上
    # train_epoch = 200
    # 保存tensorboard日志文件的位置
    # %
    if args.net == 'snn':
        net = SNNforauditory(tau=tau).to(device)
    elif args.net == 'rnn':
        net = RNNauditory().to(device)
    elif args.net == 'cnn':
        from model.TCN import TemporalConvNet

        net = TemporalConvNet(num_inputs=83 * Max_num, num_channels=[256, 11], kernel_size=2, dropout=0.5).to(device)
    print(net)
    log_dir = ''
    # 使用Adam优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=1e-4)
    # optimizer = AdaBelief(net.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.999), weight_decay=1e-2,weight_decouple=True,rectify=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    early_stopping = EarlyStopping(None,patience=20, verbose=True)

    train_times = 0
    max_test_accuracy = -1

    train_accuracy_all = []
    train_loss_all = []

    validation_accuracy_all = []
    validation_loss_all = []
    #encoder = encoding.PoissonEncoder()
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

            label_one_hot = F.one_hot(label, classnumber).float()
            # label_one_hot=label
            # label_one_hot = F.one_hot(label.unsqueeze(0).to(torch.int64), 11).float()
            # label_one_hot = smooth_labels(label_one_hot, smoothing=0.1)
            # data_seq = data.permute(1,0,2,3)
            data_seq = data.permute(3, 0, 2, 1)
            # data_seq = data.permute(1, 0, 2, 3)
            # data_seq = data_seq.unsqueeze(2)
            out_spikes_counter_frequency = 0
            # w, h = data.shape
            # data = torch.from_numpy(data)
            # out_spike = torch.full((20, w, h), 0, dtype=torch.bool)
            # T = 20
            # for t in range(T):
            # out_spike[t] = pe(result1)

            # data_per_file=out_spike.float().numpy()
            print(data.shape)
            if args.net == 'snn':
                for t in range(T):
                    # encoded_data=data[:,t,:,:]
                    # encoded_data = encoder(data)
                    # out_spikes_counter_frequency += net(encoded_data)
                    out_spikes_counter_frequency += net(data_seq[t])
                out_spikes_counter_frequency = out_spikes_counter_frequency / T
            elif args.net == 'rnn':
                out_spikes_counter_frequency = net(data_seq)
            elif args.net == 'cnn':
                out_spikes_counter_frequency = net(data_seq)
            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0

            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            # label_one_hot=label_one_hot.squeeze(0)
            # label_one_hot=label
            # loss=F.cross_entropy(out_spikes_counter_frequency, label_one_hot)
            # loss=criterion(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)
            train_losses.append(loss.item())

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label).float().sum().item()
            # train_correct_sum += (out_spikes_counter_frequency.argmax(dim=1) ==label).float().sum().item()
            train_sum += label.numel()
            train_accuracy = train_correct_sum / train_sum

            if train_times % 1 == 0:
                print('train_times:', train_times, 'train_loss:', '%.5f' % (loss.item()), 'train_accuracy:',
                      '%.5f' % train_accuracy)

            epoch_train_accuracy.append(train_accuracy)
        epoch_train_loss = np.array(train_losses).mean()
        train_loss_all.append(epoch_train_loss)
        train_accuracy_all.append(epoch_train_accuracy[-1])
        # scheduler.step()
        # 验证模型
        test_accuracy, epoch_val_loss = eval_model(net, test_loader, T)
        validation_accuracy_all.append(test_accuracy)
        validation_loss_all.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        if (test_accuracy > max_test_accuracy):
            import copy

            max_test_accuracy = test_accuracy
            best_state = {
                'net': copy.deepcopy(net.state_dict()),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }



        print(
            'dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
                train_dir, batch_size, optimizer.param_groups[0]['lr'], T, log_dir, max_test_accuracy, train_times))
        early_stopping(epoch_val_loss, net)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
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
    epochs = np.linspace(1, len(validation_accuracy_all), len(validation_accuracy_all))

    # savepath='./snnmodel/Cochlea_Coding/T10/'
    if os.path.exists(savepath):
        print('ready exist')
    else:
        print('ok I make it')
        os.makedirs(savepath)
    file_prefix = savepath + args.spec+ '_'+args.encoding+'_' +args.net+'_'+str(T) +'_'+ 'acc:%.4f' % train_accuracy_all[
        -1] + '_' + 'val_acc:%.4f' % \
                  validation_accuracy_all[-1] + '_' + str(timestamp)
    filename = file_prefix + '_train_result.csv'
    with open(filename, 'w', newline='') as file:
        fieldnames = ['epochs', 'train_accuracy_all', 'train_loss_all', 'validation_accuracy_all',
                      'validation_loss_all']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入CSV文件的表头
        writer.writeheader()
        # 写入数据
        for epochs, train_accuracy_alls, train_loss_alls, validation_accuracy_alls, validation_loss_alls in zip(epochs,
                                                                                                                np.array(
                                                                                                                    train_accuracy_all),
                                                                                                                np.array(
                                                                                                                    train_loss_all),
                                                                                                                np.array(
                                                                                                                    validation_accuracy_all),
                                                                                                                np.array(
                                                                                                                    validation_loss_all)):
            writer.writerow(
                {'epochs': epochs, 'train_accuracy_all': train_accuracy_alls, 'train_loss_all': train_loss_alls,
                 'validation_accuracy_all': validation_accuracy_alls, 'validation_loss_all': validation_loss_alls})

    modelfilename = file_prefix + '.pth'
    torch.save(best_state, modelfilename)
    net.load_state_dict(best_state['net'])
    real_test_accuracy, real_test_loss = eval_model(net, real_test_loader, T)
    with open(name_result,'a') as file:
        summary = {
            'best_acc': max(validation_accuracy_all),
            'best_loss': min(validation_loss_all),
            'real_test_acc': real_test_accuracy,
            'mode_path': filename,
        }
        writer = csv.DictWriter(file, fieldnames=summary.keys())
        # writer.writeheader()  # 写入头部，即字段名
        writer.writerow(summary)  # 写入字典作为CSV的一行

import random


def generate_random_number():
    return random.choice([-1, 0, 5, 10, 20])


# Generate a random number from the specified list
random_number = generate_random_number()
print(random_number)
