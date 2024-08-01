# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:07:31 2022

@author: CCM
"""

import numpy as np
import os
from brian2 import *
import brian2 as b2

def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # 如果是文件夹，则递归
        if os.path.isdir(file_path):
            print('进入子文件夹')
            for sub_file in os.listdir(file_path):
                sub_file_path = os.path.join(file_path, sub_file)
                if os.path.splitext(sub_file_path)[1] == file_last_name:
                    file_name.append(sub_file_path)
                else:
                    pass
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name

def get_matrix_from_file(fileName):
    readout = np.load(fileName)
    print('see readout: ', readout.shape, fileName)
    # readout : [..., [neuron i, neuron j, w], ...]
    # value_array.shape = (sensor_number x sensor_number)
    value_array = np.zeros((5, 5))
    if not readout.shape == (0,):
        # value_array[1, 1] = w1, value_array[2, 2] = w2, ......
        value_array[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]
    print(value_array)
    return value_array

txt_folder_path='/mnt/data/CCM/snndatabase/snncodingdata/'
txt_file_path = findfile(txt_folder_path, '.txt')
print(txt_file_path)

norm_data_list = []
for txt_file in txt_file_path:
    data_per_file = [[] for i in range(5)]
    with open(txt_file) as f:
        content = f.readlines()
        row = 0
        for items in content:
            row_data = items.split()
            for response_value in row_data:
                data_per_file[row].append(float(response_value))
            row += 1
    print(data_per_file)
    norm_data =np.array(data_per_file)
    norm_data_list.append(norm_data)
norm_data_list = np.array(norm_data_list)

# shuffle_indices = np.arange(norm_data_list.shape[0])
# np.random.shuffle(shuffle_indices)
# norm_data_shuffle_list = norm_data_list[shuffle_indices]

sensor_indices_copy_list = []
input_spike_times_list = []
for i in range(len(norm_data_list)):
    scaling_factor_time=1*ms
    spikes_time_matrix=norm_data_list[i]
    sensor_indices = [[] for i in range(5)]
    spike_times_list = [[] for i in range(5)]
    
    
    row = 0
    for row in range(len(spikes_time_matrix)):
        for col in range(len(spikes_time_matrix[row])):
            if spikes_time_matrix[row][col] != None:
                spike_times_list[row].append(spikes_time_matrix[row][col])
    # print(spikes_time_list)
    # 同一个神经元的输入脉冲时刻不能存在相同值
    for i in range(len(spike_times_list)):
        # 对同一个神经元的输入脉冲时刻去重
        spike_times_list[i] = spike_times_list[i]
        for j in range(len(spike_times_list[i])):
            sensor_indices[i].append(i)
    # print(sensor_indices, '\n', spike_times_list)
    # 将二维矩阵转为一维以适应嗅球模型的输入
    sensor_indices_copy = []
    spike_times_list_copy = []
    for sensor_indice in sensor_indices:
        for element in sensor_indice:
            sensor_indices_copy.append(element)
    for spike_times in spike_times_list:
        for spike_time in spike_times:
            spike_times_list_copy.append(spike_time)
    print(sensor_indices_copy, '\n', spike_times_list_copy)
    # 将每个样本的脉冲输入时刻input_spike_times和对应的神经元序号sensor_indices_copy添加到list中
    sensor_indices_copy_list.append(sensor_indices_copy)
    input_spike_times = spike_times_list_copy * scaling_factor_time
    input_spike_times_list.append(input_spike_times)
    print('sample numbers: ', len(sensor_indices_copy_list), len(input_spike_times_list))

sensor_numbers=5
run_time = 4000 * ms
input_groups = {}
neuron_groups = {}
connections = {}
state_monitors = {}
spike_monitors = {}

eqs1 = '''

    dv/dt = -v/tau : 1
    tau : second

'''

# spike generator
input_groups['input_neurons'] = SpikeGeneratorGroup(sensor_numbers, indices=[0], times=[0*ms])

# ORN
neuron_groups['AN'] = NeuronGroup(5, eqs1, threshold='v>1', reset='v = 0', method='euler')
neuron_groups['AN'].tau = 100*ms
neuron_groups['II'] = NeuronGroup(5, eqs1, threshold='v>1', reset='v = 0', method='euler')
neuron_groups['II'].tau = 300*ms
neuron_groups['IC'] = NeuronGroup(5, eqs1, threshold='v>1', reset='v = 0', method='euler')
neuron_groups['IC'].tau = 800*ms
neuron_groups['BN'] = NeuronGroup(5, eqs1, threshold='v>1', reset='v = 0', method='euler')
neuron_groups['BN'].tau = 800*ms


# define STDP connections
taupre = 20*ms
taupost= 100*ms
wmax =1
Apre = 0.0005
Apost = -Apre*taupre/taupost*1.05
eqs_stdp = '''
             w : 1
             dapre/dt = -apre/taupre : 1 (clock-driven)
             dapost/dt = -apost/taupost : 1 (clock-driven)
           '''
eqs_stdp_pre = '''
             v_post += w
             apre += Apre
             w = clip(w+apost, 0, wmax)
             '''
eqs_stdp_post = '''
             apost += Apost
             w = clip(w+apre, 0, wmax)
             '''
#%%
weight_path = '/mnt/data/CCM/snndatabase/weights/'
starting = 'sound_'
ending = '_1'
connections['S0'] = Synapses(input_groups['input_neurons'], neuron_groups['AN'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S0'].connect(j='i')
weightMatrix_S0 = get_matrix_from_file(weight_path + starting + 'S0' + ending + '.npy')
connections['S0'].w = weightMatrix_S0[connections['S0'].i, connections['S0'].j]

# ORN -> MC
connections['S1'] = Synapses(neuron_groups['AN'], neuron_groups['II'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                              on_post=eqs_stdp_post, method='exact')
connections['S1'].connect(j='i')
weightMatrix_S1 = get_matrix_from_file(weight_path + starting + 'S1' + ending + '.npy')
connections['S1'].w = weightMatrix_S1[connections['S1'].i, connections['S1'].j]


connections['S2'] = Synapses(neuron_groups['II'], neuron_groups['IC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                              on_post=eqs_stdp_post, method='exact')
connections['S2'].connect(j='i')
weightMatrix_S2 = get_matrix_from_file(weight_path + starting + 'S2' + ending + '.npy')
connections['S2'].w = weightMatrix_S2[connections['S2'].i, connections['S2'].j]

connections['S3'] = Synapses(neuron_groups['IC'], neuron_groups['BN'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                              on_post=eqs_stdp_post, method='exact')
connections['S3'].connect(condition='abs(i-j)<4 and i!=j')
weightMatrix_S3 = get_matrix_from_file(weight_path + starting + 'S3' + ending + '.npy')
connections['S3'].w = weightMatrix_S3[connections['S3'].i, connections['S3'].j]

state_monitors['M0'] = StateMonitor(neuron_groups['AN'], 'v', record=True)
spike_monitors['AN'] = SpikeMonitor(neuron_groups['AN'])
state_monitors['M1'] = StateMonitor(neuron_groups['II'], 'v', record=True)
spike_monitors['II'] = SpikeMonitor(neuron_groups['II'], variables='v')
state_monitors['M2'] = StateMonitor(neuron_groups['IC'], 'v', record=True)
spike_monitors['IC'] = SpikeMonitor(neuron_groups['IC'])
state_monitors['M3'] = StateMonitor(neuron_groups['BN'], 'v', record=True)
spike_monitors['BN'] = SpikeMonitor(neuron_groups['BN'])

spike_monitors['BN0'] = SpikeMonitor(neuron_groups['BN'][0])
spike_monitors['BN1'] = SpikeMonitor(neuron_groups['BN'][1])
spike_monitors['BN2'] = SpikeMonitor(neuron_groups['BN'][2])
spike_monitors['BN3'] = SpikeMonitor(neuron_groups['BN'][3])
spike_monitors['BN4'] = SpikeMonitor(neuron_groups['BN'][4])

state_monitors['M_S1'] = StateMonitor(connections['S1'], ['w', 'apre', 'apost'], record=True)
state_monitors['M_S2'] = StateMonitor(connections['S2'], ['w', 'apre', 'apost'], record=True)
state_monitors['M_S3'] = StateMonitor(connections['S3'], ['w', 'apre', 'apost'], record=True)
M_SN = [state_monitors['M_S1'],state_monitors['M_S2'],state_monitors['M_S3']]
#%%
net = Network()
for obj_list in [input_groups, neuron_groups, connections, state_monitors, spike_monitors]:
    for key in obj_list:
        net.add(obj_list[key])

# 先让net跑0秒是为了消除spikegeneratorgroup在0秒处的脉冲影响
net.run(0*second)

#%%
num_examples = 0
MC_spike_monitor_temp = [[] for i in range(5)]
for neuron_indices, neuron_spike_times in zip(sensor_indices_copy_list, input_spike_times_list):
    input_groups['input_neurons'].set_spikes(indices=neuron_indices, times=neuron_spike_times + num_examples * run_time)
    net.run(run_time)
    new_txt_file = open('/mnt/data/CCM/snndatabase/aftercoding/%s.txt' %
                        (txt_file_path[num_examples].split('/')[-2].split('.t')[0]+'-'+txt_file_path[num_examples].split('/')[-1].split('.t')[0]), 'w', encoding='utf-8')
    MC_spike_monitors = [spike_monitors['BN0'], spike_monitors['BN1'],spike_monitors['BN2'],spike_monitors['BN3'],spike_monitors['BN4']]
    for num_MC in range(len(MC_spike_monitors)):
        # 求出一个样本的脉冲发放时刻：net处理完该样本后新增的脉冲发放时刻
        delta_MC_spike_times = [x for x in MC_spike_monitors[num_MC].t[:] if x not in MC_spike_monitor_temp[num_MC]]
        delta_MC_spike_times -= num_examples * run_time
        print('MC spike times: ', delta_MC_spike_times)
        for MC_spike_time in delta_MC_spike_times:
            new_txt_file.writelines(str('%.4f' % (MC_spike_time/(1000*ms))) + ' ')
        new_txt_file.writelines('\n')
        MC_spike_monitor_temp[num_MC] = MC_spike_monitors[num_MC].t[:]
    new_txt_file.close()
    print("----------the network has tested %s samples----------" % (num_examples + 1))
    num_examples += 1

net.stop()
#%%
#------------------------------------------------------------------------------
# plot results
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# 查看ORN、MC和GC中的各个神经元的脉冲发放情况
plt.figure(figsize=(6, 6))
for i, name in enumerate(spike_monitors):
    # i : 0, 1, 2; name : ORN, MC, GC
    plt.subplot(len(spike_monitors), 1, 1+i)
    plt.plot(spike_monitors[name].t/(1000 * ms), spike_monitors[name].i, '.')
    plt.title('Spikes of population ' + name)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.figure(figsize=(12, 6))
num = 1
# 查看脉冲发放频率最高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(2, 3, num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[0], label='w, ie 0-1')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    else:
        plt.subplot(2, 3, num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[0], label='w, ee 0-0')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    num += 1

num = 1
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(2, 3, 3 + num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[1], label='w, ie 1-0')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    else:
        plt.subplot(2, 3, 3 + num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[1], label='w, ee 1-1')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    num += 1
plt.subplots_adjust(wspace=0.25, hspace=0.5)

plt.show()
#%%
test_name='/mnt/data/CCM/snndatabase/snncodingdata/1/44.txt'
print('/mnt/data/CCM/snndatabase/aftercoding/%s.txt' %
                        (test_name.split('/')[-2].split('.t')[0]+'-'+test_name.split('/')[-1].split('.t')[0]))
#%%
import brian2
brian2.test()