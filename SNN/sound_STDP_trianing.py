# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:39:13 2022

@author: CCM
"""

#%%
import numpy as np
import os
from brian2 import *
import brian2genn
set_device('genn')
# prefs.devices.cpp_standalone.openmp_threads = 32

#%%
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
#%%
txt_folder_path='/mnt/data/CCM/snndatabase/snncodingdata/'
txt_file_path = findfile(txt_folder_path, '.txt')
print(txt_file_path)
#%%
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

shuffle_indices = np.arange(norm_data_list.shape[0])
np.random.shuffle(shuffle_indices)
norm_data_shuffle_list = norm_data_list[shuffle_indices]
#%%
sensor_indices_copy_list = []
input_spike_times_list = []
for i in range(len(norm_data_list)):
    scaling_factor_time=1*ms
    spikes_time_matrix=norm_data_shuffle_list[i]
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
#%%

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


neuron_groups['AC'] = NeuronGroup(5, eqs1, threshold='v>1', reset='v = 0', method='euler')
neuron_groups['AC'].tau = 800*ms



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


# input -> ORN
connections['S0'] = Synapses(input_groups['input_neurons'], neuron_groups['AN'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S0'].connect(j='i')
connections['S0'].w = 0.4

# ORN -> MC
connections['S1'] = Synapses(neuron_groups['AN'], neuron_groups['II'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                              on_post=eqs_stdp_post, method='exact')
connections['S1'].connect(j='i')
connections['S1'].w = 0.5


connections['S2'] = Synapses(neuron_groups['II'], neuron_groups['IC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                              on_post=eqs_stdp_post, method='exact')
connections['S2'].connect(j='i')
connections['S2'].w = 0.5


connections['S3'] = Synapses(neuron_groups['IC'], neuron_groups['AC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                              on_post=eqs_stdp_post, method='exact')
connections['S3'].connect(condition='abs(i-j)<4 and i!=j')
connections['S3'].w = 0.5

# # MC -> GC
# connections['S2'] = Synapses(neuron_groups['MC'], neuron_groups['GC'], model=eqs_stdp, on_pre=eqs_stdp_pre_ei,
#                              on_post=eqs_stdp_post_ei, method='exact')
# connections['S2'].connect(j='i')
# connections['S2'].w = 600

# # GC -> MC
# connections['S3'] = Synapses(neuron_groups['GC'], neuron_groups['MC'], model=eqs_stdp, on_pre=eqs_stdp_pre_ie,
#                              on_post=eqs_stdp_post_ie, method='exact')
# connections['S3'].connect(condition='i!=j')
# connections['S3'].w = -800
# # visualise_connectivity(connections['S3'])

state_monitors['M0'] = StateMonitor(neuron_groups['AN'], 'v', record=True)
spike_monitors['AN'] = SpikeMonitor(neuron_groups['AN'])
state_monitors['M1'] = StateMonitor(neuron_groups['II'], 'v', record=True)
spike_monitors['II'] = SpikeMonitor(neuron_groups['II'], variables='v')
state_monitors['M2'] = StateMonitor(neuron_groups['IC'], 'v', record=True)
spike_monitors['IC'] = SpikeMonitor(neuron_groups['IC'])
state_monitors['M3'] = StateMonitor(neuron_groups['AC'], 'v', record=True)
spike_monitors['AC'] = SpikeMonitor(neuron_groups['AC'])
state_monitors['M_S1'] = StateMonitor(connections['S1'], ['w', 'apre', 'apost'], record=True)
state_monitors['M_S2'] = StateMonitor(connections['S2'], ['w', 'apre', 'apost'], record=True)
state_monitors['M_S3'] = StateMonitor(connections['S3'], ['w', 'apre', 'apost'], record=True)
M_SN = [state_monitors['M_S1'],state_monitors['M_S2'],state_monitors['M_S3']]

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------
#%%
net = Network()
for obj_list in [input_groups, neuron_groups, connections, state_monitors, spike_monitors]:
    for key in obj_list:
        net.add(obj_list[key])

# 先让net跑0秒是为了消除spikegeneratorgroup在0秒处的脉冲影响
net.run(0*second)

num_examples = 0
for neuron_indices, neuron_spike_times in zip(sensor_indices_copy_list, input_spike_times_list):
    input_groups['input_neurons'].set_spikes(indices=neuron_indices, times=neuron_spike_times + num_examples * run_time)
    net.run(run_time)

    print("----------the network has trained %s samples----------" % (num_examples + 1))
    num_examples += 1

net.stop()
#%%
#------------------------------------------------------------------------------
# save weight results
#------------------------------------------------------------------------------

print('save weight results...')
starting = 'sound_'
ending = '_1'
for connName in connections:
    print('saving ' + connName + '...')
    conn = connections[connName]
    # 将zip数据封装为list(zip())
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save('/mnt/data/CCM/snndatabase/weights_1/' + starting + connName + ending, connListSparse)
print('saving weights finished')

#%%
#------------------------------------------------------------------------------
# plot results
#------------------------------------------------------------------------------

# 查看ORN、MC和GC中的各个神经元的脉冲发放情况
figure(figsize=(6, 6))
for i, name in enumerate(spike_monitors):
    # i : 0, 1, 2; name : ORN, MC, GC
    subplot(len(spike_monitors), 1, 1+i)
    plot(spike_monitors[name].t[int(39089/2):int(39089/2)+500]/(1000 * ms), spike_monitors[name].i[int(39089/2):int(39089/2)+500], '|',color='k')
    title(name+' spike' )
    axis('off')

subplots_adjust(wspace=1, hspace=0.5)
show()
#%%
print(len(spike_monitors[name].t[int(39089/2):int(39089/2)+500]/(1000 * ms)))
print(len(spike_monitors[name].i[:]))
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(18, 9))
num = 1
# 查看脉冲发放频率最高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(3, 3, num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[0],color='r',linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num==1:
            plt.title('AN->II_Synapse1')
        if num==2:
            plt.title('II->IC_Synapse1')
        if num==3:
            plt.title('IC->AC_Synapse1')
    else:
        plt.subplot(3, 3, num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[0],color='r',linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num==1:
            plt.title('AN->II_Synapse1')
        if num==2:
            plt.title('II->IC_Synapse1')
        if num==3:
            plt.title('IC->AC_Synapse1')
    num += 1

num = 1
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(3, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[1],color='g',linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num==1:
            plt.title('AN->II_Synapse2')
        if num==2:
            plt.title('II->IC_Synapse2')
        if num==3:
            plt.title('IC->AC_Synapse2')
    else:
        plt.subplot(3, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[1],color='g',linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num==1:
            plt.title('AN->II_Synapse2')
        if num==2:
            plt.title('II->IC_Synapse2')
        if num==3:
            plt.title('IC->AC_Synapse2')
    num += 1

num = 1
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(3, 3, 6 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[4],color='b',linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num==1:
            plt.title('AN->II_Synapse3')
        if num==2:
            plt.title('II->IC_Synapse3')
        if num==3:
            plt.title('IC->AC_Synapse3')
    else:
        plt.subplot(3, 3, 6 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[4],color='b',linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num==1:
            plt.title('AN->II_Synapse3')
        if num==2:
            plt.title('II->IC_Synapse3')
        if num==3:
            plt.title('IC->AC_Synapse3')
    num += 1

plt.subplots_adjust(wspace=0.25, hspace=0.5)
plt.show()

num = 1
#%%
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(2, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[2], label='w, ie 1-0')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    else:
        plt.subplot(2, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[2], label='w, ee 1-1')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    num += 1
num = 1
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(2, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[3], label='w, ie 1-0')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    else:
        plt.subplot(2, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[3], label='w, ee 1-1')
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.title('Synapse_%s' % num)
    num += 1
plt.subplots_adjust(wspace=0.25, hspace=0.5)

plt.show()
#%%
# plt.figure()
plt.figure(figsize=(8, 9))
plt.subplot(4,1,1)
# plt.plot(spike_monitors['AN'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5),spike_monitors['AN'].count/(run_time*len(sensor_indices_copy_list)), tick_label=['1', '2', '3', '4', '5'], width=0.3)
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('AN Firing rate')
plt.ylim([0,12])
plt.subplot(4,1,2)
# plt.plot(spike_monitors['II'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5),spike_monitors['II'].count/(run_time*len(sensor_indices_copy_list)), tick_label=['1', '2', '3', '4', '5'], width=0.3,color='r')
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('II Firing rate')
plt.ylim([0,12])
plt.subplot(4,1,3)
# plt.plot(spike_monitors['IC'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5),spike_monitors['IC'].count/(run_time*len(sensor_indices_copy_list)), tick_label=['1', '2', '3', '4', '5'], width=0.3,color='g')
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('IC Firing rate')
plt.ylim([0,12])
plt.subplot(4,1,4)
# plt.plot(spike_monitors['AC'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5),spike_monitors['AC'].count/(run_time*len(sensor_indices_copy_list)), tick_label=['1', '2', '3', '4', '5'], width=0.3,color='b')
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('AC Firing rate')
plt.ylim([0,12])
plt.subplots_adjust(wspace=0.1, hspace=0.7)
plt.show()
#%%
plt.figure()
plt.plot(range(5))
plt.show()
#%%
figure()
plot(range(N),spike_monitors['B_CH'].count/4000*ms)
xlabel('neuron index')
ylabel('Firing rate (sp/s)');
#%%
from brian2 import *
import brian2genn
set_device('genn')
n = 1000
duration = 1*second
tau = 10*ms
eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''
group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='exact')
group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'
monitor = SpikeMonitor(group)
run(duration)