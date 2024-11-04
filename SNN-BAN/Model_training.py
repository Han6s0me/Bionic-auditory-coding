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
import sys
from collections import deque
import scipy.signal
import pyaudio
import struct as st


parser = ArgumentParser()
parser.add_argument('--spec', type=str, default='cqt', help='spectrogram type')
parser.add_argument('--net', type=str, default='snn', help='network type')
parser.add_argument('--encoding', type=str, default='BAN', help='encoding model')

args = parser.parse_args()


train_dir = '.snndatabase/TID_train_8k/' #loading training data
test_dir='.snndatabase/TID_test_8k/' #loading test data


Max_num = 83
T = 10
batch_size = 64
learning_rate = 1e-3
tau = 2.0
train_epoch = 200
classnumber = 11
savepath = '' #save model path



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):

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
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  
        self.val_loss_min = val_loss


import numpy as np


def pre_emphasis(signal, coefficient=0.97):

    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])
    return emphasized_signal


def normalize_audio_peak(audio, target_peak):

    current_peak = max(abs(audio))

    scale = target_peak / current_peak

    normalized_audio = audio * scale

    return normalized_audio





def ShortTimeEnergy(signal, windowLength, step):
    
    signal = signal / np.max(signal) 
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

    Hist = np.histogram(C, bins=10)  # bins=10 3
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3)  
    if len(MaximaC) >= 2:
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

            random_number = -1
            if random_number == -1:
                sounddata = sounddata
            else:
                sounddata = add_noise(sounddata, random_number)
            sounddata = pre_emphasis(sounddata,coefficient=0.95)

            segments = VAD(sounddata, 8000)
            if len(segments) != 0:
                # sounddata = sounddata[segments[0][0]:segments[-1][1]]
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
                stft_result = librosa.core.stft(y=sounddata, n_fft=164, hop_length=96)
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
            # TEST.append(temp)
        else:
            temp = train_data[idx][:, 0:Max_num]
        TEST.append(temp)

    # test=np.array(TEST)
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
  

    import numpy as np
    if args.encoding=='BAN':
        file = './Encoding_model/BAN_a_b_c_d_list.txt'
        parameter_list = np.zeros((83, 80))
        f = open(file, 'r')
        content = f.readlines()
        f.close()

        row = 0
        for items in content:
            data_i = items.split()
            print(row)
            idx = 0
            for x in data_i:
                parameter_list[row][idx] = x
                idx += 1
            row += 1

        train_data = BAN_encoding(train_data, parameter_list)
        real_test_data = BAN_encoding(real_test_data, parameter_list)
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



    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron,surrogate, functional, layer
    import torch.nn.functional as F


    def smooth_labels(labels, smoothing=0.1):
        num_classes = labels.shape[1]

        smooth_value = smoothing / num_classes
        with torch.no_grad():
            smooth_labels = labels * (1 - smoothing) + smooth_value
        return smooth_labels


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


    class RNNauditory(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(input_size=83 * Max_num, hidden_size=128,num_layers=2, batch_first=False)
            self.fc = nn.Linear(128, classnumber, bias=True)

        def forward(self, x: torch.Tensor):
            outputs, (hn, cn) = self.rnn(x.flatten(2))
            out = self.fc(hn[0])
            out = torch.softmax(out, dim=-1)
            return out


    def eval_model(net, test_loader, T):
        net.eval()
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

            print('Epoch %s' % (epoch + 1), 'test_loss:', '%.5f' % epoch_val_loss, 'test_accuracy:',
                  '%.5f' % test_accuracy)
            return test_accuracy, epoch_val_loss

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=True)
    real_test_loader = DataLoader(
        TensorDataset(torch.from_numpy(real_test_data).float(), torch.from_numpy(real_test_labels)),
        batch_size=batch_size, shuffle=True)
    test_loader = real_test_loader


    device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    print('Total GPU numbers:' + str(torch.cuda.device_count()), '\n', 'Being uesd GPU:' + str(device))
    torch.cuda.manual_seed(0)

    if args.net == 'snn':
        net = SNNforauditory(tau=tau).to(device)

    elif args.net == 'rnn':
        net = RNNauditory().to(device)
    elif args.net == 'cnn':
        from TCN.TCN import TemporalConvNet

        net = TemporalConvNet(num_inputs=83 * Max_num, num_channels=[128, classnumber], kernel_size=2, dropout=0.5).to(device)
    print(net)
    print('total params: %.2fM' % (get_params(net) / 1e6))
    log_dir = ''
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
  
    early_stopping = EarlyStopping(None,patience=20, verbose=True)

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
            train_times += 1 
            data = data.to(device)

            label = label.long().to(device)

            label_one_hot = F.one_hot(label, classnumber).float()

            data_seq = data.permute(3, 0, 2, 1)
   
            out_spikes_counter_frequency = 0
    
            print(data.shape)
            if args.net == 'snn':
                for t in range(T):
    
                    out_spikes_counter_frequency += net(data_seq[t])
                out_spikes_counter_frequency = out_spikes_counter_frequency / T
            elif args.net == 'rnn':
                out_spikes_counter_frequency = net(data_seq)
            elif args.net == 'cnn':
                out_spikes_counter_frequency = net(data_seq)

            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
   
            loss.backward()
            optimizer.step()

            functional.reset_net(net)
            train_losses.append(loss.item())

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
        if early_stopping.early_stop:
            print("Early stopping")
            break  
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

    epochs = np.linspace(1, len(validation_accuracy_all), len(validation_accuracy_all))
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

        writer.writeheader()
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
        writer.writerow(summary)  

