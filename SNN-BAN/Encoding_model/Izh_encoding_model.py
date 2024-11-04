#%%
from scipy import signal
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import librosa
from librosa import cqt
import sys
from collections import deque
import scipy.signal
import pyaudio
import struct as st
from scipy.io import savemat

# Choose Datasets

train_dir = '../snndatabase/TID_DATA_8k_all/'
train_dir = '../snndatabase/RWCP_train_8k_all/'
test_dir = '../snndatabase/TID_TEST_8k_all/'
test_dir = '../snndatabase/RWCP_test_8k_all/'

# Encoding parameter
T = 10  # timesteps

# Speech pre-emphasis
def pre_emphasis(signal, coefficient=0.95):

    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])

    return emphasized_signal

# Normalization amplitude
def normalize_audio_peak(audio, target_peak):

    current_peak = max(abs(audio))
    scale = target_peak / current_peak
    normalized_audio = audio * scale

    return normalized_audio


# Voice endpoint detection
# Calculate short-time energy
def ShortTimeEnergy(signal, windowLength, step):
    
    signal = signal / np.max(signal)  
    curPos = 0
    L = len(signal)
    numOfFrames = np.asarray(np.floor((L - windowLength) / step) + 1, dtype=int)
    E = np.zeros((numOfFrames, 1))
    for i in range(numOfFrames):
        window = signal[int(curPos):int(curPos + windowLength - 1)];
        E[i] = (1 / (windowLength)) * np.sum(np.abs(window ** 2));
        curPos = curPos + step;

    return E

# Calculate spectral centroid
def SpectralCentroid(signal, windowLength, step, fs):

    signal = signal / np.max(signal) 
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

# Finding local maxima
def findMaxima(f, step):

    countMaxima = 0
    Maxima = []
    for i in range(len(f) - step - 1):  
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
    Weight = 100  # Parameters of threshold estimation

    # Finding the threshold of short-term energy
    Hist = np.histogram(E, bins=10)  
    HistE = Hist[0]
    X_E = Hist[1]
    MaximaE, countMaximaE = findMaxima(HistE, 3)  
    if len(MaximaE) >= 2:  
        T_E = (Weight * X_E[MaximaE[0][0]] + X_E[MaximaE[1][0]]) / (Weight + 1)
    else:
        T_E = E_mean / 2

    # Threshold for finding spectral centroid
    Hist = np.histogram(C, bins=10) #bins=10 3
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3) #3 10
    if len(MaximaC) >= 2:
        T_C = (Weight * X_C[MaximaC[0][0]] + X_C[MaximaC[1][0]]) / (Weight + 1)
    else:
        T_C = Z_mean / 2

    # Threshold judgment
    Flags1 = (E >= T_E)
    Flags2 = (C >= T_C)
    flags = np.array(Flags1 & Flags2, dtype=int)

    ## Extracting speech fragments
    count = 1
    segments = []
    while count < len(flags):  
        # 初始化
        curX = []
        countTemp = 1
        while ((flags[count - 1] == 1) and (count < len(flags))):
            if countTemp == 1:  
                Limit1 = np.round((count - 1) * step * fs) + 1 
                if Limit1 < 1:
                    Limit1 = 1
            count = count + 1  
            countTemp = countTemp + 1  

        if countTemp > 1:  
            Limit2 = np.round((count - 1) * step * fs)  
            if Limit2 > len(signal):
                Limit2 = len(signal)
    
            segments.append([int(Limit1), int(Limit2)])
        count = count + 1

    for i in range(len(segments) - 1): 
        if segments[i][1] >= segments[i + 1][0]:
            segments[i][1] = segments[i + 1][1]
            segments[i + 1, :] = []
            i = 1

    return segments

# Finding datasets file
def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        
        if os.path.isdir(file_path):
            findfile(file_path, file_last_name)
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name

# Adding noise
Noise, sr_noise = librosa.load('../snndatabase/noise92/babble.wav', sr=19980)
def add_noise(sound,snr,noise=Noise):
    old_sample_rate = 19980
    SAMPLE_RATE = 8000
    noise = librosa.resample(noise, orig_sr=old_sample_rate, target_sr=SAMPLE_RATE)
    min_length = min(len(sound), len(noise))
    start_index = np.random.randint(0, len(noise) - len(sound) + 1)
    sound = sound[:min_length]

    noise = noise[start_index:start_index + len(sound)]

    signal_energy = np.sum(sound ** 2)
    noise_energy = np.sum(noise ** 2)

    target_snr_db = snr  # 目标SNR为10dB
    target_snr_linear = 10 ** (target_snr_db / 10)  
    mixing_ratio = np.sqrt(signal_energy / (target_snr_linear * noise_energy))

    mixed_audio = sound + mixing_ratio * noise
    return mixed_audio

# Loading datasets file and extract features
def read_origindata(dir):  
   
    folder_nums = 0  
    file_nums_count = 0  
    data = []  
    labels = []  
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        print(folder_nums, folder, folder_path)  
        file_path = findfile(folder_path, '.wav')
        label = folder.split('_')[0]
        for file in file_path:

            waveform, sample_rate = torchaudio.load(file)
            sounddata = waveform.t().numpy()
            sounddata = sounddata.reshape(-1, )
        
            #sounddata = add_noise(sounddata, 0) # Whether to add noise SNR=0
            sounddata=pre_emphasis(sounddata)

            segments = VAD(sounddata, 8000)
            if len(segments) != 0:
                if len(segments) != 0:
                differences = [max(sublist) - min(sublist) for sublist in segments]
                max_diff_index = differences.index(max(differences))
                list_with_max_diff = segments[max_diff_index]
                sounddata = sounddata[list_with_max_diff[0]:list_with_max_diff[1]]
            sounddata = normalize_audio_peak(sounddata, 1)
            cqtpec = cqt(sounddata, sr=sample_rate, fmin=32, n_bins=83, hop_length=96)
      
            cqtm, phase = librosa.core.magphase(cqtpec)
            Engry = cqtm

            Min_data = np.min(Engry)
            Max_data = np.max(Engry)
            data_per_file = (Engry - Min_data) / (Max_data - Min_data)
 
            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
       
        folder_nums += 1

    labels = np.array(labels)
    return data, labels

train_data, train_labels = read_origindata(train_dir)

# Normalization
Max_num=83
TEST=[]
for idx in range(len(train_data)):
    if Max_num > train_data[idx].shape[1]:
       zeronum=abs(Max_num-train_data[idx].shape[1]);
       temp=np.pad(train_data[idx],((0,0),(0,zeronum)))
       # TEST.append(temp)
    else:
       temp = train_data[idx][:, 0:Max_num]
    TEST.append(temp)

#test=np.array(TEST)
test=np.stack(TEST, axis=0)
train_data=test

# Izh encoding
def lzhikevich_model(T,I, a, b, c, d):
    V_Statue=[]
    v = -65  
    u = b * v
    Timepoint=[]
    spikes = 0
    for idx in range(T):  
        v +=  (0.04 * v ** 2 + 5 * v + 140 - u + I)
        u += a * (b * v - u)  

        if v >= 30:  
            v = c  
            u += d  
            Timepoint.append(idx)
        V_Statue.append(v)

    return np.array(Timepoint),np.array(V_Statue)


TEST=[]
for idx in range(train_data.shape[0]):
    data_per_file = train_data[idx, :, :]
    F_data = []
    T_data = []
    for time in range(data_per_file.shape[1]):
        T_temp = []
        for frq in range(data_per_file.shape[0]):
            Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 2)
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
train_data=test123

# #%% Save data  Switch the save path according to the datasets. For excample ../TID/.. or ../RWCP/..
# savemat('../After_encoding_data/TID/Izh_encoding/train_data_10T.mat', {'train_data': train_data})
# savemat('../After_encoding_data/TID/Izh_encoding/label_data_10T.mat', {'train_labels': train_labels})
# savemat('../After_encoding_data/TID/Izh_encoding/TEST/test_data_10T.mat', {'train_data': Test_data})
# savemat('../After_encoding_data/TID/Izh_encoding/TEST/labeltest_data_10T.mat', {'train_labels': test_labels})