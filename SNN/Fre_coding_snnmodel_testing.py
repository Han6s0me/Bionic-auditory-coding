#%% 批量测试
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
test_dir = '/mnt/data/CCM/snndatabase/TID_TEST_8k_all/'
# test_dir='/mnt/data/CCM/snndatabase/RWCP_test_8k/'
T=10
# modelpath='/home/handsome/PythonProject/SNN/snnmodel/Freq_Coding/T2/Max_num:203_acc:1.0000_val_acc:0.6964.pth'
# modelpath='/home/handsome/PythonProject/SNN/snnmodel/ccm_snn_model_TID_cqt_possion_T5_1.pth'
# % 音频缩放
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
# Noise, sr_noise = librosa.load('/mnt/data/CCM/snndatabase/noise92/babble.wav', sr=8000)
Noise, sr_noise = librosa.load('/mnt/data/CCM/snndatabase/noise92/babble.wav', sr=19980)
def add_noise(sound,snr,noise=Noise):
    old_sample_rate=19980
    SAMPLE_RATE=8000
    noise = librosa.resample(noise, orig_sr=old_sample_rate, target_sr=SAMPLE_RATE)
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
            sounddata = pre_emphasis(sounddata)
            segments = VAD(sounddata, 8000)

            if len(segments) != 0:
                sounddata = sounddata[segments[0][0]:segments[-1][1]]

            sounddata = normalize_audio_peak(sounddata, 1)


            # f, t, Sxx = signal.spectrogram(sounddata, sample_rate)
            # Engry=10*np.log10(Sxx)
            cqtpec = cqt(sounddata, sr=sample_rate, fmin=32, n_bins=83, hop_length=96)
            # Engry = abs(cqtpec)
            cqtm, phase = librosa.core.magphase(cqtpec)
            Engry = cqtm
            # sounddata=torch.from_numpy(sounddata).type(torch.FloatTensor)
            # x1=torch.stft(sounddata,n_fft=128,hop_length=96,win_length=128,window=torch.hann_window(128),onesided=True)
            # ttt=torch.sqrt(torch.sum(x1**2,dim=-1))
            # Engry=ttt.numpy()
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

# test_dir=r'D:\浙大项目\SNN\Database\DATA\DATA_WAV'
# test_dir=r'D:\浙大项目\SNN\TID_TEST_8k'
# test_dir = '/mnt/data/CCM/snndatabase/TID_TEST_8k/'
test_data, test_labels = read_origindata(test_dir)
# % 补长
# Max_num=92
#%
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
encoder = encoding.PoissonEncoder()
# T=10
TEST=[]
for idx in range(test_data.shape[0]):
# for idx in range(1):
    data_per_file = test_data[idx, :, :]
    x = torch.from_numpy(data_per_file)
    w, h = x.shape
    out_spike = torch.full((T, w, h), 0, dtype=torch.bool)
    for t in range(T):
        out_spike[t] = encoder(x)
    TEST.append(np.array(out_spike))
    print(idx)
test123=np.stack(TEST, axis=0)
# test_data=test123
Test_data=test123
# #%%
# from scipy.io import savemat
# temp = torch.from_numpy(Test_data).type(torch.FloatTensor)
# Test_data=temp.permute(0,3,2,1)
# Test_data=np.array(Test_data)
# Test_data = np.squeeze(Test_data)
# savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Possion_coding/TEST/test_data_10T.mat', {'train_data': Test_data})
# savemat('/home/handsome/PythonProject/SNN/snnmodel/After_coding_data/Possion_coding/TEST/labeltest_data_10T.mat', {'train_labels': test_labels})
#
# #%%
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





modelname=findfile('/home/handsome/PythonProject/SNN/snnmodel/Model_TID_R_MSE/Freq_encoding/T10/','.pth')
#%
accuracyScore_ALL = []
macro_precisionScore_ALL = []
macro_recallScore_ALL = []
macro_f1Score_ALL = []
macro_AUC_ALL = []
for idx in range(len(modelname)):
    # modelpath='/home/handsome/PythonProject/SNN/snnmodel/Resultmodel_Freq_Coding_RWCP/T20/Max_num:83_acc:1.0000_val_acc:0.9500_1699001536.pth'
    modelpath=modelname[idx]
    savepath='/home/handsome/PythonProject/SNN/snnmodel/Model_TID_R_MSE/Freq_encoding/T10/'
    # % 计算
    # net = torch.load('./ccm_snn_model_Arua.pth',map_location='cpu')
    net = torch.load(modelpath, map_location='cpu')
    # net = SNNforauditory(tau=4.0)
    # net.load_state_dict(torch.load('./snnmodel/ccm_snn_model_TID_cqt_possion_T5_2.pth'))
    net.eval()
    with torch.inference_mode():


        SUM = 0
        pred = []
        pred_probabilities = []
        for idx in range(Test_data.shape[0]):
            test_output = 0
            result1 = Test_data[idx, :, :, :]
            result1 = result1[np.newaxis, :]
            result1 = torch.from_numpy(result1).type(torch.FloatTensor)
            result1 = result1.permute(1, 0, 2, 3)

            # encoder = encoding.PoissonEncoder()
            for t in range(T):
                # encoded_data = encoder(result1)
                # test_output += net(encoded_data)
                # net.eval()
                test_output += net(result1[t])
            test_output = test_output / T

            Label = np.argmax(test_output.numpy())
            pred.append(Label)
            pred_probabilities.append(torch.softmax(test_output, dim=1).numpy())
            if Label == test_labels[idx]:
                SUM = SUM + 1
            print(idx)
        print(SUM / Test_data.shape[0])
    # %
    if os.path.exists(savepath):
        print('ready exist')
    else:
        print('ok I make it')
        os.makedirs(savepath)
    pred_label = np.array(pred)
    true_label = np.array(test_labels)
    figurename=str.split(str.split(modelpath,'/')[-1],'.pth')[0]
    pred_probabilities = np.array(pred_probabilities).squeeze()
    # from sklearn.metrics import confusion_matrix
    #
    # #%
    # # def confusion(y_label, y_pred):
    # # con_mat = confusion_matrix(y_label.astype(str), y_pred.astype(str))
    # con_mat = confusion_matrix(true_label.astype(str), pred_label.astype(str))
    # # print(con_mat)
    # classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10']
    # # classes.sort()
    # plt.imshow(con_mat, cmap=plt.cm.Blues)
    # indices = range(len(con_mat))
    # plt.xticks(indices, classes)
    # plt.yticks(indices, classes)
    # plt.colorbar()
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # for first_index in range(len(con_mat)):
    #     for second_index in range(len(con_mat[first_index])):
    #         plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    # plt.savefig(savepath + figurename + '_ACC:' + str(SUM / Test_data.shape[0]) + '_1' + '.tif', dpi=300)
    # plt.show()
    #
    #
    # # confusion(true_label, pred_label)
    #
    # # %
    # cm = confusion_matrix(true_label, pred_label)
    # plt.imshow(cm, cmap=plt.cm.Blues)
    # plt.colorbar()
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.xticks(np.arange(11))
    # plt.yticks(np.arange(11))
    # plt.savefig(savepath+figurename+'_ACC:'+str(SUM /Test_data.shape[0])+'_2'+'.tif', dpi=300)
    # plt.show()
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    from sklearn.metrics import classification_report
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import csv


    #
    # # confusion_matrix 混淆矩阵 及 相关推演出来的指标
    def cal_metrics(y_test, y_predict):

        accuracyScore = accuracy_score(y_test, y_predict)
        # matrix = confusion_matrix(y_test, y_predict)

        # weigthed_precisionScore = precision_score(y_test, y_predict,average='weighted')
        # micro_precisionScore = precision_score(y_test, y_predict, average='micro')
        macro_precisionScore = precision_score(y_test, y_predict, average='macro')

        # weigthed_recallScore = recall_score(y_test, y_predict,average='weighted')
        # micro_recallScore = recall_score(y_test, y_predict, average='micro')
        macro_recallScore = recall_score(y_test, y_predict, average='macro')

        # weigthed_f1Score = f1_score(y_test, y_predict,average='weighted')
        # micro_f1Score = f1_score(y_test, y_predict,average='micro')
        macro_f1Score = f1_score(y_test, y_predict, average='macro')
        n_classes = np.unique(y_test).size
        y_true_binary = label_binarize(y_test, classes=range(n_classes))

        # weigthed_AUC = roc_auc_score(y_true_binary, pred_probabilities, average='weighted', multi_class='ovo')
        # micro_AUC = roc_auc_score(y_test, pred_probabilities,average='micro',multi_class='ovo')
        macro_AUC = roc_auc_score(y_true_binary, pred_probabilities, average='macro', multi_class='ovo')

        # report = classification_report(y_test, y_predict)

        print("accuracy_score = {}".format(accuracyScore))
        # print("confusion_matrix = {}".format(matrix))

        # print("weigthed_precisionScore = {}".format(weigthed_precisionScore))
        # print("micro_precisionScore = {}".format(micro_precisionScore))
        print("macro_precisionScore = {}".format(macro_precisionScore))

        # print("weigthed_recallScore  = {}".format(weigthed_recallScore ))
        # print("micro_recallScore  = {}".format(micro_recallScore))
        print("macro_recallScore  = {}".format(macro_recallScore))

        # print("weigthed_f1Score = {}".format(weigthed_f1Score))
        # print("micro_f1Score = {}".format(micro_f1Score))
        print("macro_f1Score = {}".format(macro_f1Score))

        # print(report)
        # print("weigthed_AUC = {}".format(weigthed_AUC))
        # print("micro_AUC = {}".format(micro_AUC))
        print("macro_AUC = {}".format(macro_AUC))

        return accuracyScore, macro_precisionScore, macro_recallScore, macro_f1Score, macro_AUC


    accuracyScore, macro_precisionScore, macro_recallScore, macro_f1Score, macro_AUC = cal_metrics(true_label,
                                                                                                   pred_label)

    accuracyScore_ALL.append(accuracyScore)
    macro_precisionScore_ALL.append(macro_precisionScore)
    macro_recallScore_ALL.append(macro_recallScore)
    macro_f1Score_ALL.append(macro_f1Score)
    macro_AUC_ALL.append(macro_AUC)

    filename = savepath + 'model_result.csv'
    final_name = final_name = str.split(figurename, '_')
    with open(filename, 'w', newline='') as file:
        fieldnames = ['name', 'accuracyScore_all', 'macro_precisionScore_all', 'macro_recallScore_all',
                      'macro_f1Score_all', 'macro_AUC_all']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入CSV文件的表头
        writer.writeheader()
        for accuracyScore_alls, macro_precisionScore_alls, macro_recallScore_alls, macro_f1Score_alls, macro_AUC_alls in zip(
                np.array(accuracyScore_ALL), np.array(macro_precisionScore_ALL), np.array(macro_recallScore_ALL),
                np.array(macro_f1Score_ALL), np.array(macro_AUC_ALL)):
            writer.writerow({'name': final_name[-1], 'accuracyScore_all': accuracyScore_alls,
                             'macro_precisionScore_all': macro_precisionScore_alls,
                             'macro_recallScore_all': macro_recallScore_alls,
                             'macro_f1Score_all': macro_f1Score_alls,
                             'macro_AUC_all': macro_AUC_alls})