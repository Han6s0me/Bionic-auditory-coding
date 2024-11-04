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
parser.add_argument('--encoding', type=str, default='BAN', help='encoding model')
parser.add_argument('--snr', type=str, default='SNR', help='snr')


args = parser.parse_args()

train_dir = './snndatabase/TID_train_8k/'
test_dir = './snndatabase/TID_test_8k/'

Max_num = 83
T = 10
batch_size = 64
learning_rate = 1e-3
tau = 2.0
train_epoch = 100
classnumber=11
savepath = ''
def pre_emphasis(signal, coefficient=0.97):
   
    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])
    return emphasized_signal


def normalize_audio_peak(audio, target_peak):
    
    current_peak = max(abs(audio))

    
    scale = target_peak / current_peak

    normalized_audio = audio * scale

    return normalized_audio




import sys
from collections import deque

import scipy.signal
import pyaudio
import struct as st


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


def findMaxima(f, step):
    
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
    Weight = 100  
   
    Hist = np.histogram(E, bins=10)  
    HistE = Hist[0]
    X_E = Hist[1]
    MaximaE, countMaximaE = findMaxima(HistE, 3) 
    if len(MaximaE) >= 2:  
        T_E = (Weight * X_E[MaximaE[0][0]] + X_E[MaximaE[1][0]]) / (Weight + 1)
    else:
        T_E = E_mean / 2

    
    Hist = np.histogram(C, bins=10)  
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3)  
        T_C = (Weight * X_C[MaximaC[0][0]] + X_C[MaximaC[1][0]]) / (Weight + 1)
    else:
        T_C = Z_mean / 2

    
    Flags1 = (E >= T_E)
    Flags2 = (C >= T_C)
    flags = np.array(Flags1 & Flags2, dtype=int)

    
    count = 1
    segments = []
    while count < len(flags): 
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


def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        if os.path.isdir(file_path):
            findfile(file_path, file_last_name)
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name


Noise, sr_noise = librosa.load('./snndatabase/noise92/babble.wav', sr=19980)


def add_noise(sound, snr, noise=Noise):
    old_sample_rate = 19980
    SAMPLE_RATE = 8000
    noise = librosa.resample(noise, orig_sr=old_sample_rate, target_sr=SAMPLE_RATE)

    sound = sound / max(abs(sound))
    noise = noise / max(abs(noise))
    sound = sound /1
    noise = noise /1

    min_length = min(len(sound), len(noise))
    sound = sound[:min_length]
   
    noise = noise[:min_length]
    signal_energy = np.sum(sound ** 2)
    noise_energy = np.sum(noise ** 2)

    
    target_snr_db = snr  
    target_snr_linear = 10 ** (target_snr_db / 10)  
    mixing_ratio = np.sqrt(signal_energy / (target_snr_linear * noise_energy))

    
    mixed_audio = sound + mixing_ratio * noise
    return mixed_audio

def read_origindata(dir,snr_number=-1):  
    
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
           
            if snr_number == -1:
                sounddata = sounddata
            else:
                sounddata = add_noise(sounddata, snr_number)
            sounddata = pre_emphasis(sounddata,coefficient=0.95)
            segments = VAD(sounddata, 8000)
            if len(segments) != 0:
             
                differences = [max(sublist) - min(sublist) for sublist in segments]
                max_diff_index = differences.index(max(differences))
                list_with_max_diff = segments[max_diff_index]
                sounddata = sounddata[list_with_max_diff[0]:list_with_max_diff[1]]
            sounddata = normalize_audio_peak(sounddata, 1)
            if 'cqt' in args.spec:
                print('Using constant-Q transform')
                cqtpec = cqt(sounddata, sr=sample_rate, fmin=32, n_bins=83, hop_length=96)
                cqtm, phase = librosa.core.magphase(cqtpec)
                Engry = cqtm
    
            elif args.spec == 'mel2':
                print("Using mel spectrogram")
                mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=192,
                    hop_length=96,
                    n_mels=83
                )

                Engry = mel_spectrogram(torch.from_numpy(sounddata)).numpy()
                
            elif args.spec == 'stft':
                print("Using naive stft")
                stft_result = librosa.stft(y=sounddata, n_fft=164, hop_length=96)
                magnitude, phase = librosa.magphase(stft_result)
               
                Engry = magnitude

            Min_data = np.min(Engry)
            Max_data = np.max(Engry)
            data_per_file = (Engry - Min_data) / (Max_data - Min_data)
          
            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
        folder_nums += 1
   

    labels = np.array(labels)
    return data, labels
def data_pad(train_data):
  
    TEST = []
    for idx in range(len(train_data)):
        if Max_num > train_data[idx].shape[1]:
            zeronum = abs(Max_num - train_data[idx].shape[1]);
            temp = np.pad(train_data[idx], ((0, 0), (0, zeronum)))
        else:
            temp = train_data[idx][:, 0:Max_num]
        TEST.append(temp)

    test = np.stack(TEST, axis=0)
    train_data = test
    return train_data


def lzhikevich_model(T, I, a, b, c, d):
    V_Statue = []
    v = -65  
    u = b * v
    Timepoint = []

    spikes = 0
    for idx in range(T): 
        v += (0.04 * v ** 2 + 5 * v + 140 - u + I)
        u += a * (b * v - u)  

        if v >= 30: 
            v = c  
            u += d  
            Timepoint.append(idx)
        V_Statue.append(v)

    return np.array(Timepoint), np.array(V_Statue)
def BAN_encoding(train_data, parameter_list):
    TEST = []
    for idx in range(train_data.shape[0]):
        data_per_file = train_data[idx, :, :]
        F_data = []
        T_data = []
        for time in range(data_per_file.shape[1]):
            T_temp = []
            for frq in range(data_per_file.shape[0]):
               
                temp = int(data_per_file[frq][time] * 20)
                if temp == 20:
                    temp = 19
                Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, parameter_list[frq, temp],
                                                       parameter_list[frq, temp + 20],
                                                       parameter_list[frq, temp + 40], parameter_list[frq, temp + 60])

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
                Timepoint, V_Statue = lzhikevich_model(T, data_per_file[frq][time] * 100, 0.02, 0.2, -65, 2)

                spike_array = np.zeros(T, dtype=bool)
                for i in Timepoint:
                    spike_array[i] = True
                T_temp = F_data.append(spike_array)
                pass
            T_temp = np.stack(F_data, axis=0)
                   T_data.append(T_temp)
            F_data = []
            T_temp = []
        test = np.stack(T_data, axis=0)
        TEST.append(test)
        print(idx)

    test123 = np.stack(TEST, axis=0)
    train_data = test123
    return train_data

class LIFNeuron:
            def __init__(self, membrane_resistance=10, membrane_time_scale=8, firing_threshold=1):
                self.membrane_resistance = membrane_resistance  
                self.membrane_time_scale = membrane_time_scale  
                self.firing_threshold = firing_threshold 
                self.membrane_potential = 0  

            def simulate(self, input_current, time):
               
                dt = 1  
                time_steps = int(time / dt)  
                membrane_potential_trace = []  
                Timepoint = []

               
                for idx in range(time_steps):
                    
                    dV = dt * (-(
                        self.membrane_potential) + self.membrane_resistance * input_current) / self.membrane_time_scale
                    self.membrane_potential += dV

                   
                    if self.membrane_potential >= self.firing_threshold:
                        membrane_potential_trace.append(self.firing_threshold)  
                        self.membrane_potential = 0  
                        Timepoint.append(idx)
                    else:
                        membrane_potential_trace.append(self.membrane_potential) 

               
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
    temp = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_data = temp.permute(0, 3, 2, 1)
    train_data = np.array(train_data)
 
    return train_data

encoding_cache_path_snr = os.path.join(savepath, args.spec+ '-' +args.encoding+ '-' +str(T) +'-' +args.snr+'-encoding_cache.npy')
if os.path.exists(encoding_cache_path_snr):
    print('Loading encoding cache from', encoding_cache_path_snr)
    encoding_cache = np.load(encoding_cache_path_snr, allow_pickle=True).item()

    real_test_data_snr0 = encoding_cache['real_test_data_snr0']
    real_test_labels_snr0 = encoding_cache['real_test_labels_snr0']
    real_test_data_snr5 = encoding_cache['real_test_data_snr5']
    real_test_labels_snr5 = encoding_cache['real_test_labels_snr5']
    real_test_data_snr10 = encoding_cache['real_test_data_snr10']
    real_test_labels_snr10 = encoding_cache['real_test_labels_snr10']
    real_test_data_snr20 = encoding_cache['real_test_data_snr20']
    real_test_labels_snr20 = encoding_cache['real_test_labels_snr20']
else:
    print('Generating encoding cache to', encoding_cache_path_snr)
    real_test_data_snr0, real_test_labels_snr0 = read_origindata(test_dir,0)
    real_test_data_snr5, real_test_labels_snr5 = read_origindata(test_dir, 5)
    real_test_data_snr10, real_test_labels_snr10 = read_origindata(test_dir, 10)
    real_test_data_snr20, real_test_labels_snr20 = read_origindata(test_dir, 20)

    real_test_data_snr0 = data_pad(real_test_data_snr0)
    real_test_data_snr5 = data_pad(real_test_data_snr5)
    real_test_data_snr10 = data_pad(real_test_data_snr10)
    real_test_data_snr20 = data_pad(real_test_data_snr20)
    import numpy as np

    if args.encoding == 'BAN':
       
        file = './Encoding_model/CQT_a_b_c_d_list.txt'

        parameter_list = np.zeros((83, 80))
         f = open(file, 'r')
        content = f.readlines()
        f.close()
        # parameter_list=np.array(content)
        row = 0
        for items in content:
            data_i = items.split()
            print(row)
            idx = 0
            for x in data_i:
                parameter_list[row][idx] = x
                idx += 1
            row += 1

        real_test_data_snr0 = BAN_encoding(real_test_data_snr0, parameter_list)
        real_test_data_snr5 = BAN_encoding(real_test_data_snr5, parameter_list)
        real_test_data_snr10 = BAN_encoding(real_test_data_snr10, parameter_list)
        real_test_data_snr20 = BAN_encoding(real_test_data_snr20, parameter_list)
    
    elif args.encoding == 'LIF':


        # real_test_data_snr = LIF_encoding(real_test_data_snr)
        real_test_data_snr0 = LIF_encoding(real_test_data_snr0)
        real_test_data_snr5 = LIF_encoding(real_test_data_snr5)
        real_test_data_snr10 = LIF_encoding(real_test_data_snr10)
        real_test_data_snr20 = LIF_encoding(real_test_data_snr20)
    elif args.encoding == 'IZH':

        # real_test_data_snr = IZH_encoding(real_test_data_snr)
        real_test_data_snr0 = IZH_encoding(real_test_data_snr0)
        real_test_data_snr5 = IZH_encoding(real_test_data_snr5)
        real_test_data_snr10 = IZH_encoding(real_test_data_snr10)
        real_test_data_snr20 = IZH_encoding(real_test_data_snr20)
    elif args.encoding == 'POISSON':

        # real_test_data_snr = POISSON_encoding(real_test_data_snr)
        real_test_data_snr0 = POISSON_encoding(real_test_data_snr0)
        real_test_data_snr5 = POISSON_encoding(real_test_data_snr5)
        real_test_data_snr10 = POISSON_encoding(real_test_data_snr10)
        real_test_data_snr20 = POISSON_encoding(real_test_data_snr20)

    np.save(encoding_cache_path_snr,
            { 'real_test_data_snr0': real_test_data_snr0,
             'real_test_labels_snr0': real_test_labels_snr0,
              'real_test_data_snr5': real_test_data_snr5,
              'real_test_labels_snr5': real_test_labels_snr5,
              'real_test_data_snr10': real_test_data_snr10,
              'real_test_labels_snr10': real_test_labels_snr10,
              'real_test_data_snr20': real_test_data_snr20,
              'real_test_labels_snr20': real_test_labels_snr20})


from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
name_result=savepath + 'summary_'  + args.spec+ '-'+args.encoding+'-' +args.net+'-'+str(T)+'-'+args.snr + '.csv'

with open(name_result, 'a') as file:
    summary = {
        'real_test_acc':0,
        'mode_path': 0,
        'mode_snr0':0,
        'mode_snr5': 0,
        'mode_snr10': 0,
        'mode_snr20': 0
    }
    writer = csv.DictWriter(file, fieldnames=summary.keys())
    writer.writeheader()  

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class SNNforauditory(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(83, 64, kernel_size=9 ,stride=1, padding=4),  
            nn.MaxPool1d(kernel_size=3),
            neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
         
        )
       

        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            nn.Linear(64 * (Max_num // 3), 1024, bias=False),
            neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
            layer.Dropout(0.5),
           

        )
        self.fc1 = nn.Sequential(
           
            nn.Linear(1024, classnumber, bias=False),       
            neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),


        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)       
        x = self.fc(x)     
        x = self.fc1(x)
        
        return x


device = torch.device('cpu')
import torch.nn.functional as F
def eval_model(net, test_loader, T):
    net.eval()
    # with torch.no_grad():
    with torch.inference_mode():
      
        test_sum = 0
        correct_sum = 0
        val_losses = []
        for test_data, test_label in test_loader:
            test_data = test_data.to(device)
            test_label = test_label.long().to(device)
            test_label_one_hot = F.one_hot(test_label, classnumber).float()
           
            test_data_seq = test_data.permute(3, 0, 2, 1)
            
            test_output = 0
           
            if args.net == 'snn':
                for t in range(T):  
                    test_output += net(test_data_seq[t])
                test_output = test_output / T
            elif args.net == 'rnn':
                test_output = net(test_data_seq)
            elif args.net == 'cnn':
                test_output = net(test_data_seq)

            val_loss = F.mse_loss(test_output, test_label_one_hot)
           
            val_losses.append(val_loss.item())
            correct_sum += (test_output.max(1)[1] == test_label).float().sum().item()  
            test_sum += test_label.numel()  
           
            functional.reset_net(net)

        epoch_val_loss = np.array(val_losses).mean()
        test_accuracy = correct_sum / test_sum

        return test_accuracy, epoch_val_loss

from torch.utils.data import Dataset, DataLoader, TensorDataset
real_test_loader_snr0 = DataLoader(
    TensorDataset(torch.from_numpy(real_test_data_snr0).float(), torch.from_numpy(real_test_labels_snr0)),
    batch_size=batch_size, shuffle=False)
real_test_loader_snr5 = DataLoader(
    TensorDataset(torch.from_numpy(real_test_data_snr5).float(), torch.from_numpy(real_test_labels_snr5)),
    batch_size=batch_size, shuffle=False)
real_test_loader_snr10 = DataLoader(
    TensorDataset(torch.from_numpy(real_test_data_snr10).float(), torch.from_numpy(real_test_labels_snr10)),
    batch_size=batch_size, shuffle=False)
real_test_loader_snr20 = DataLoader(
    TensorDataset(torch.from_numpy(real_test_data_snr20).float(), torch.from_numpy(real_test_labels_snr20)),
    batch_size=batch_size, shuffle=False)

import pandas as pd
import os
cav_result=savepath + 'summary_'  + args.spec+ '-'+args.encoding+'-' +args.net+'-'+str(T) + '.csv'

main_df = pd.read_csv(cav_result)
csv_files = main_df['mode_path']
acc_result = main_df['real_test_acc']
all_data = []
idx=0

for file in csv_files:
    if file.endswith('.csv'):
        new_file = file.replace('_train_result.csv', '.pth')
        net = SNNforauditory(tau=2.0)
        state_dict = torch.load(new_file, map_location='cpu')
        net.load_state_dict(state_dict['net'])

        real_test_accuracy_snr0, real_test_loss_snr0 = eval_model(net, real_test_loader_snr0, T)
        real_test_accuracy_snr5, real_test_loss_snr5 = eval_model(net, real_test_loader_snr5, T)
        real_test_accuracy_snr10, real_test_loss_snr10 = eval_model(net, real_test_loader_snr10, T)
        real_test_accuracy_snr20, real_test_loss_snr20 = eval_model(net, real_test_loader_snr20, T)
        with open(name_result, 'a') as file:
            summary = {
                'real_test_acc':acc_result[idx],
                'mode_path': new_file,
                'mode_snr0':real_test_accuracy_snr0,
                'mode_snr5': real_test_accuracy_snr5,
                'mode_snr10': real_test_accuracy_snr10,
                'mode_snr20': real_test_accuracy_snr20
            }
            writer = csv.DictWriter(file, fieldnames=summary.keys())
            writer.writerow(summary)  

    idx+=1
    print('Finish:',idx)