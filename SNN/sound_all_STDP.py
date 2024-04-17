# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:39:13 2022

@author: CCM
"""

# %%
import numpy as np
import os
from brian2 import *
import brian2genn
defaultclock.dt = 1*ms
# set_device('genn')


# prefs.devices.cpp_standalone.openmp_threads = 32

# %%
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


# %%
# txt_folder_path = '/mnt/data/CCM/snndatabase/snncodingdata_short/'
# txt_folder_path = '/mnt/data/CCM/snndatabase/snncodingdata_18N_5s/'
txt_folder_path = '/mnt/data/CCM/snndatabase/snncodingdata_36N_5s/'

# txt_folder_path = '/mnt/data/CCM/snndatabase/RWCP_DATA/RWCP_train/'
txt_file_path = findfile(txt_folder_path, '.txt')
print(txt_file_path)
# %%
norm_data_list = []
for txt_file in txt_file_path:
    # data_per_file = [[] for i in range(5)]
    data_per_file = [[] for i in range(36)]
    with open(txt_file) as f:
        content = f.readlines()
        row = 0
        for items in content:
            row_data = items.split()
            for response_value in row_data:
                data_per_file[row].append(float(response_value))
            row += 1
    print(data_per_file)
    norm_data = np.array(data_per_file)
    norm_data_list.append(norm_data)
norm_data_list = np.array(norm_data_list)

shuffle_indices = np.arange(norm_data_list.shape[0])
np.random.shuffle(shuffle_indices)
norm_data_shuffle_list = norm_data_list[shuffle_indices]
# %%
sensor_indices_copy_list = []
input_spike_times_list = []
for i in range(len(norm_data_list)):
    scaling_factor_time = 1 * ms
    spikes_time_matrix = norm_data_shuffle_list[i]
    # sensor_indices = [[] for i in range(5)]
    # spike_times_list = [[] for i in range(5)]
    sensor_indices = [[] for i in range(36)]
    spike_times_list = [[] for i in range(36)]

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

    # c = list(zip(sensor_indices_copy_list, input_spike_times_list))
    # shuffle(c)
    # sensor_indices_copy_list,input_spike_times_list=zip(*c)

# %%

# sensor_numbers = 5
# run_time = 4000 * ms
sensor_numbers = 36
sensor_numbers_V=180

run_time = 6000 * ms
input_groups = {}
neuron_groups = {}
connections = {}
state_monitors = {}
spike_monitors = {}

# eqs1 = '''
#
#     dv/dt = -v/tau : 1
#     tau : second
#
# '''
taum = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -70*mV
er=-70*mV
# El=-60*mV
taue = 5*ms
taui=10*ms
eqs1 = '''
dv/dt = (-gi*(v-er)+ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
dgi/dt = -gi / taui : 1
'''


# spike generator
input_groups['input_neurons'] = SpikeGeneratorGroup(sensor_numbers, indices=[0], times=[0 * ms])

# ORN
neuron_groups['AN'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['AN'].tau = 50 * ms
# neuron_groups['AN'].tau = 50 * ms
neuron_groups['AN'].v=El

neuron_groups['SON'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['SON'].tau = 800 * ms
# neuron_groups['SON'].tau = 50 * ms
neuron_groups['SON'].v=El

neuron_groups['II'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['II'].tau = 800 * ms
# neuron_groups['II'].tau = 50 * ms
neuron_groups['II'].v=El

neuron_groups['IC'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['IC'].tau = 800 * ms
# neuron_groups['IC'].tau = 50 * ms
neuron_groups['IC'].v=El

neuron_groups['THA'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['THA'].tau = 800 * ms
# neuron_groups['THA'].tau = 50 * ms
neuron_groups['THA'].v=El

neuron_groups['AC_CORE'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['AC_CORE'].tau = 800 * ms
# neuron_groups['AC_CORE'].tau = 300 * ms
neuron_groups['AC_CORE'].v=El

neuron_groups['AC_BELT'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['AC_BELT'].tau = 800 * ms
# neuron_groups['AC_BELT'].tau = 300 * ms
neuron_groups['AC_BELT'].v=El

neuron_groups['AC_PARA'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['AC_PARA'].tau = 800 * ms
# neuron_groups['AC_PARA'].tau = 300 * ms
neuron_groups['AC_PARA'].v=El

neuron_groups['OFC'] = NeuronGroup(10, eqs1, threshold='v>vt', reset='v = vr', method='euler')
# neuron_groups['OFC'].tau = 800 * ms
# neuron_groups['OFC'].tau = 300 * ms
neuron_groups['OFC'].v=El


neuron_groups['INIB_OFC'] = NeuronGroup(10, eqs1, threshold='v>vt', reset='v = vr',method='euler')
# neuron_groups['OFC'].tau = 800 * ms
# neuron_groups['OFC'].tau = 300 * ms
neuron_groups['INIB_OFC'].v=El

neuron_groups['INIB_AC_CORE'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr',method='euler')
# neuron_groups['OFC'].tau = 800 * ms
# neuron_groups['OFC'].tau = 300 * ms
neuron_groups['INIB_AC_CORE'].v=El

neuron_groups['INIB_AC_BELT'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr',method='euler')
# neuron_groups['OFC'].tau = 800 * ms
# neuron_groups['OFC'].tau = 300 * ms
neuron_groups['INIB_AC_BELT'].v=El

neuron_groups['INIB_AC_PARA'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>vt', reset='v = vr',method='euler')
# neuron_groups['OFC'].tau = 800 * ms
# neuron_groups['OFC'].tau = 300 * ms
neuron_groups['INIB_AC_PARA'].v=El
# define STDP connections
# taupre = 20 * ms
# taupost = 20 * ms
# wmax = 1
# Apre = 0.01
# Apost = -Apre * taupre / taupost * 2.05
# eqs_stdp = '''
#              w : 1
#              dapre/dt = -apre/taupre : 1 (clock-driven)
#              dapost/dt = -apost/taupost : 1 (clock-driven)
#            '''
# eqs_stdp_pre = '''
#              v_post += w
#              apre += Apre*(1-w)
#              w = clip(w+apost, 0, wmax)
#              '''
# eqs_stdp_post = '''
#              apost += Apost*w
#              w = clip(w+apre, 0, wmax)
#              '''

taupre = 20*ms
taupost = taupre
gmax = 1
dApre = .0001
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax
eqs_stdp = '''
             w : 1
             dApre/dt = -Apre / taupre : 1 (event-driven)
             dApost/dt = -Apost / taupost : 1 (event-driven)
           '''
eqs_stdp_pre = '''
             ge += w
             Apre += dApre
             w = clip(w + Apost, 0, gmax)
             '''
eqs_stdp_post = '''
             Apost += dApost
             w = clip(w + Apre, 0, gmax)
             '''


tau_stdp = 20*ms
eqs_stdp_inhib = '''
    w : 1
    dApre/dt=-Apre/tau_stdp : 1 (event-driven)
    dApost/dt=-Apost/tau_stdp : 1 (event-driven)
'''

alpha = 3*Hz*tau_stdp*2  # Target rate parameter
gmax_i = 0.1         # Maximum inhibitory weight
eta = 1e-2
on_pre_i = '''Apre += 1.
            w = clip(w+(Apost)*eta, 0, gmax_i)
            gi += w'''
on_post_i='''Apost += 1.
          w = clip(w+Apre*eta, 0, gmax_i)
        '''
# input -> ORN
# connections['SI'] = Synapses(input_groups['input_neurons'], neuron_groups['AN'], model='''w : 1''',
#                              on_pre='''v += w''', method='exact')
# # connections['SI'].connect(j='i')
# connections['SI'].connect('abs(i-j)>=(5*i) and abs(i-j)<=(5*(i+1))')
# connections['SI'].w = 0.25


connections['SI'] = Synapses(input_groups['input_neurons'], neuron_groups['AN'],model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['SI'].connect(j='i')
connections['SI'].connect('abs(i-j)>=(5*i) and abs(i-j)<=(5*(i+1))')
# connections['SI'].w = 0.5
connections['SI'].w = 'rand()* gmax'

# ORN -> MC
connections['S0'] = Synapses(neuron_groups['AN'], neuron_groups['SON'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
connections['S0'].connect(j='i')
# connections['S0'].connect(p=0.5)
# connections['S0'].connect('abs(i-j)>=(5*i) and abs(i-j)<=(5*(i+1))')
connections['S0'].w = 'rand()* gmax'
# connections['S0'].w = 0.5

connections['S1'] = Synapses(neuron_groups['SON'], neuron_groups['II'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
connections['S1'].connect(j='i')
# connections['S1'].connect(p=0.1)
# connections['S1'].w = 0.5
connections['S1'].w = 'rand()* gmax'

connections['S2'] = Synapses(neuron_groups['II'], neuron_groups['IC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
connections['S2'].connect(j='i')
# connections['S2'].connect(p=0.1)
# connections['S2'].w = 0.5
connections['S2'].w = 'rand()* gmax'

# connections['S3'] = Synapses(neuron_groups['AN'], neuron_groups['IC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
#                              on_post=eqs_stdp_post, method='euler')
# connections['S3'].connect(j='i')
# # connections['S3'].connect(p=0.1)
# # connections['S3'].w = 0.5
# connections['S3'].w = 'rand()* gmax'

# connections['S4'] = Synapses(neuron_groups['SON'], neuron_groups['IC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
#                              on_post=eqs_stdp_post, method='euler')
# connections['S4'].connect(j='i')
# # connections['S4'].connect(p=0.1)
# # connections['S4'].w = 0.5
# connections['S4'].w = 'rand()* gmax'

connections['S5'] = Synapses(neuron_groups['IC'], neuron_groups['THA'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
connections['S5'].connect(j='i')
# connections['S5'].connect(p=0.1)
# connections['S5'].w = 0.5
connections['S5'].w = 'rand()* gmax'

connections['S6'] = Synapses(neuron_groups['THA'], neuron_groups['AC_CORE'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
connections['S6'].connect(j='i')
# connections['S6'].connect(p=0.01)
# connections['S6'].w = 0.5
connections['S6'].w = 'rand()* gmax'

connections['S6_i'] = Synapses(neuron_groups['AC_CORE'], neuron_groups['INIB_AC_CORE'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S6'].connect(j='i')
connections['S6_i'].connect(j='i')
# connections['S6'].w = 0.5
connections['S6_i'].w = 0.5*gmax

connections['S6_i_back'] = Synapses(neuron_groups['INIB_AC_CORE'], neuron_groups['AC_CORE'], model=eqs_stdp_inhib, on_pre=on_pre_i,
                             on_post=on_post_i, method='euler')
# connections['S6'].connect(j='i')
# connections['S6_i_back'].connect(condition='i!=j')
connections['S6_i_back'].connect(p=1)
# connections['S6'].w = 0.5
connections['S6_i_back'].w = 0.5*gmax_i



connections['S7'] = Synapses(neuron_groups['AC_CORE'], neuron_groups['AC_BELT'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S7'].connect(condition='abs(i-j)<4 and i!=j')
connections['S7'].connect(p=0.01)
# connections['S7'].w = 0.5
# connections['S7'].w = 0.5
connections['S7'].w = 'rand()* gmax'

connections['S7_i'] = Synapses(neuron_groups['AC_BELT'], neuron_groups['INIB_AC_BELT'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S6'].connect(j='i')
connections['S7_i'].connect(j='i')
# connections['S6'].w = 0.5
connections['S7_i'].w = 0.5*gmax

connections['S7_i_back'] = Synapses(neuron_groups['INIB_AC_BELT'], neuron_groups['AC_BELT'], model=eqs_stdp_inhib, on_pre=on_pre_i,
                             on_post=on_post_i, method='euler')
# connections['S6'].connect(j='i')
# connections['S7_i_back'].connect(condition='i!=j')
connections['S7_i_back'].connect(p=1)
# connections['S6'].w = 0.5
connections['S7_i_back'].w =0.5*gmax_i

connections['S8'] = Synapses(neuron_groups['AC_BELT'], neuron_groups['AC_PARA'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S8'].connect(condition='abs(i-j)<4 and i!=j')
connections['S8'].connect(p=0.01)
# connections['S8'].w = 0.5
connections['S8'].w = 'rand()* gmax'

connections['S9'] = Synapses(neuron_groups['AC_BELT'], neuron_groups['OFC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S9'].connect(condition='abs(i-j)<4 and i!=j')
connections['S9'].connect(p=0.01)
# connections['S9'].w = 0.5
connections['S9'].w = 'rand()* gmax'

connections['S10'] = Synapses(neuron_groups['THA'], neuron_groups['AC_BELT'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S10'].connect(j='i')
connections['S10'].connect(p=0.01)
# connections['S10'].w = 0.5
connections['S10'].w = 'rand()* gmax'

connections['S11'] = Synapses(neuron_groups['THA'], neuron_groups['AC_PARA'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S11'].connect(j='i')
connections['S11'].connect(p=0.01)
# connections['S11'].w = 0.5
connections['S11'].w = 'rand()* gmax'

connections['S11_i'] = Synapses(neuron_groups['AC_PARA'], neuron_groups['INIB_AC_PARA'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S6'].connect(j='i')
connections['S11_i'].connect(j='i')
# connections['S6'].w = 0.5
connections['S11_i'].w = 0.5*gmax

connections['S11_i_back'] = Synapses(neuron_groups['INIB_AC_PARA'], neuron_groups['AC_PARA'], model=eqs_stdp_inhib, on_pre=on_pre_i,
                             on_post=on_post_i,method='euler')
# connections['S6'].connect(j='i')
# connections['S11_i_back'].connect(condition='i!=j')
connections['S11_i_back'].connect(p=1)
# connections['S6'].w = 0.5
connections['S11_i_back'].w = 0.5*gmax_i


connections['S12'] = Synapses(neuron_groups['AC_PARA'], neuron_groups['OFC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S12'].connect(condition='abs(i-j)<4 and i!=j')
connections['S12'].connect(p=0.01)
# connections['S12'].w = 0.5
connections['S12'].w = 'rand()* gmax'


connections['S12_i'] = Synapses(neuron_groups['OFC'], neuron_groups['INIB_OFC'], model=eqs_stdp, on_pre=eqs_stdp_pre,
                             on_post=eqs_stdp_post, method='euler')
# connections['S6'].connect(j='i')
connections['S12_i'].connect(j='i')
# connections['S6'].w = 0.5
connections['S12_i'].w = 0.5*gmax

connections['S12_i_back'] = Synapses(neuron_groups['INIB_OFC'], neuron_groups['OFC'], model=eqs_stdp_inhib, on_pre=on_pre_i,
                             on_post=on_post_i,method='euler')
# connections['S6'].connect(j='i')
# connections['S12_i_back'].connect(condition='i!=j')
connections['S12_i_back'].connect(p=1)
# connections['S6'].w = 0.5
connections['S12_i_back'].w = 0.5*gmax_i



# connections['S13'] = Synapses(neuron_groups['OFC'], neuron_groups['OFC'], model='''w : 1''',
#                              on_pre='''v -= w''', method='exact')
# # connections['S12'].connect(condition='abs(i-j)<4 and i!=j')
# connections['S13'].connect(p=1)
# # connections['S12'].w = 0.5
# connections['S13'].w = 1

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

# state_monitors['M0'] = StateMonitor(neuron_groups['AN'], 'v', record=True)
spike_monitors['AN'] = SpikeMonitor(neuron_groups['AN'])

# state_monitors['M1'] = StateMonitor(neuron_groups['SON'], 'v', record=True)
spike_monitors['SON'] = SpikeMonitor(neuron_groups['SON'])

# state_monitors['M2'] = StateMonitor(neuron_groups['II'], 'v', record=True)
spike_monitors['II'] = SpikeMonitor(neuron_groups['II'])

# state_monitors['M3'] = StateMonitor(neuron_groups['IC'], 'v', record=True)
spike_monitors['IC'] = SpikeMonitor(neuron_groups['IC'])

# state_monitors['M4'] = StateMonitor(neuron_groups['THA'], 'v', record=True)
spike_monitors['THA'] = SpikeMonitor(neuron_groups['THA'])

# state_monitors['M5'] = StateMonitor(neuron_groups['AC_CORE'], 'v', record=True)
spike_monitors['AC_CORE'] = SpikeMonitor(neuron_groups['AC_CORE'])

# state_monitors['M6'] = StateMonitor(neuron_groups['AC_BELT'], 'v', record=True)
spike_monitors['AC_BELT'] = SpikeMonitor(neuron_groups['AC_BELT'])

# state_monitors['M7'] = StateMonitor(neuron_groups['AC_PARA'], 'v', record=True)
spike_monitors['AC_PARA'] = SpikeMonitor(neuron_groups['AC_PARA'])

state_monitors['M8'] = StateMonitor(neuron_groups['OFC'], 'v', record=True)
spike_monitors['OFC'] = SpikeMonitor(neuron_groups['OFC'])


spike_monitors['AC_CORE_INIB']=SpikeMonitor(neuron_groups['INIB_AC_CORE'])
spike_monitors['AC_BELT_INIB']=SpikeMonitor(neuron_groups['INIB_AC_BELT'])
spike_monitors['AC_PARA_INIB']=SpikeMonitor(neuron_groups['INIB_AC_PARA'])
spike_monitors['OFC_INIB']=SpikeMonitor(neuron_groups['INIB_OFC'])


# state_monitors['M_SI'] = StateMonitor(connections['SI'], ['w', 'apre', 'apost'], record=True)

state_monitors['M_S0'] = StateMonitor(connections['S0'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S1'] = StateMonitor(connections['S1'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S2'] = StateMonitor(connections['S2'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
# state_monitors['M_S3'] = StateMonitor(connections['S3'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
# state_monitors['M_S4'] = StateMonitor(connections['S4'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S5'] = StateMonitor(connections['S5'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S6'] = StateMonitor(connections['S6'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S7'] = StateMonitor(connections['S7'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S8'] = StateMonitor(connections['S8'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S9'] = StateMonitor(connections['S9'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S10'] = StateMonitor(connections['S10'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S11'] = StateMonitor(connections['S11'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)
state_monitors['M_S12'] = StateMonitor(connections['S12'], ['w', 'Apre', 'Apost'], record=True,dt=1000*ms)

M_SN = [state_monitors['M_S0'], state_monitors['M_S1'],state_monitors['M_S2'],
         state_monitors['M_S5'],state_monitors['M_S6'], state_monitors['M_S7'],
        state_monitors['M_S8'], state_monitors['M_S9'],state_monitors['M_S10'], state_monitors['M_S11'],state_monitors['M_S12']]

# ------------------------------------------------------------------------------
# run the simulation and set inputs
# ------------------------------------------------------------------------------
 # %%
net = Network()
for obj_list in [input_groups, neuron_groups, connections, state_monitors, spike_monitors]:
    for key in obj_list:
        net.add(obj_list[key])

# 先让net跑0秒是为了消除spikegeneratorgroup在0秒处的脉冲影响
net.run(0 * second)

num_examples = 0
for neuron_indices, neuron_spike_times in zip(sensor_indices_copy_list, input_spike_times_list):
    input_groups['input_neurons'].set_spikes(indices=neuron_indices, times=neuron_spike_times + num_examples * (run_time))
    net.run(run_time)
    print("----------the network has trained %s samples----------" % (num_examples + 1))
    num_examples += 1
    if num_examples==5:
        break
net.stop()
# %%
# ------------------------------------------------------------------------------
# save weight results
# ------------------------------------------------------------------------------

print('save weight results...')
starting = 'sound_'
ending = '_1'
for connName in connections:
    print('saving ' + connName + '...')
    conn = connections[connName]
    # 将zip数据封装为list(zip())
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save('/mnt/data/CCM/snndatabase/weights_18N_5s_backup/' + starting + connName + ending, connListSparse)
print('saving weights finished')

# %%
# ------------------------------------------------------------------------------
# plot results
# ------------------------------------------------------------------------------

# 查看ORN、MC和GC中的各个神经元的脉冲发放情况
figure(figsize=(20, 20))
for i, name in enumerate(spike_monitors):
    # i : 0, 1, 2; name : ORN, MC, GC
    subplot(len(spike_monitors), 1, 1 + i)
    plot(spike_monitors[name].t[:] / (1000 * ms),
         spike_monitors[name].i[:], '|', color='k')
    title(name + ' spike')
    axis('off')

subplots_adjust(wspace=1, hspace=0.5)
show()
# %%
print(len(spike_monitors[name].t[int(39089 / 2):int(39089 / 2) + 500] / (1000 * ms)))
print(len(spike_monitors[name].i[:]))
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 9))
num = 1
# 查看脉冲发放频率最高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(3, 3, num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[0], color='r', linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num == 1:
            plt.title('AN->II_Synapse1')
        if num == 2:
            plt.title('II->IC_Synapse1')
        if num == 3:
            plt.title('IC->AC_Synapse1')
    else:
        plt.subplot(3, 3, num)
        plt.plot(M_Sn.t / (1000 * ms), M_Sn.w[0], color='r', linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num == 1:
            plt.title('AN->II_Synapse1')
        if num == 2:
            plt.title('II->IC_Synapse1')
        if num == 3:
            plt.title('IC->AC_Synapse1')
    num += 1

num = 1
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(3, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[1], color='g', linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num == 1:
            plt.title('AN->II_Synapse2')
        if num == 2:
            plt.title('II->IC_Synapse2')
        if num == 3:
            plt.title('IC->AC_Synapse2')
    else:
        plt.subplot(3, 3, 3 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[1], color='g', linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num == 1:
            plt.title('AN->II_Synapse2')
        if num == 2:
            plt.title('II->IC_Synapse2')
        if num == 3:
            plt.title('IC->AC_Synapse2')
    num += 1

num = 1
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        plt.subplot(3, 3, 6 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[4], color='b', linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num == 1:
            plt.title('AN->II_Synapse3')
        if num == 2:
            plt.title('II->IC_Synapse3')
        if num == 3:
            plt.title('IC->AC_Synapse3')
    else:
        plt.subplot(3, 3, 6 + num)
        plt.plot(M_Sn.t[:] / (1000 * ms), M_Sn.w[4], color='b', linewidth=2)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        if num == 1:
            plt.title('AN->II_Synapse3')
        if num == 2:
            plt.title('II->IC_Synapse3')
        if num == 3:
            plt.title('IC->AC_Synapse3')
    num += 1

plt.subplots_adjust(wspace=0.25, hspace=0.5)
plt.show()

num = 1
# %%
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
# %%
# plt.figure()
plt.figure(figsize=(8, 9))
plt.subplot(4, 1, 1)
# plt.plot(spike_monitors['AN'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5), spike_monitors['AN'].count / (run_time * len(sensor_indices_copy_list)),
        tick_label=['1', '2', '3', '4', '5'], width=0.3)
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('AN Firing rate')
plt.ylim([0, 12])
plt.subplot(4, 1, 2)
# plt.plot(spike_monitors['II'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5), spike_monitors['II'].count / (run_time * len(sensor_indices_copy_list)),
        tick_label=['1', '2', '3', '4', '5'], width=0.3, color='r')
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('II Firing rate')
plt.ylim([0, 12])
plt.subplot(4, 1, 3)
# plt.plot(spike_monitors['IC'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5), spike_monitors['IC'].count / (run_time * len(sensor_indices_copy_list)),
        tick_label=['1', '2', '3', '4', '5'], width=0.3, color='g')
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('IC Firing rate')
plt.ylim([0, 12])
plt.subplot(4, 1, 4)
# plt.plot(spike_monitors['AC'].count/(run_time*len(sensor_indices_copy_list)))
plt.bar(range(5), spike_monitors['AC'].count / (run_time * len(sensor_indices_copy_list)),
        tick_label=['1', '2', '3', '4', '5'], width=0.3, color='b')
plt.xlabel('neuron index')
plt.ylabel('Firing rate (sp/s)');
plt.title('AC Firing rate')
plt.ylim([0, 12])
plt.subplots_adjust(wspace=0.1, hspace=0.7)
plt.show()
# %%
plt.figure()
plt.plot(range(5))
plt.show()
# %%
figure()
plot(range(N), spike_monitors['B_CH'].count / 4000 * ms)
xlabel('neuron index')
ylabel('Firing rate (sp/s)');
# %%
figure(figsize=(20, 40))
for i, name in enumerate(spike_monitors):
    # i : 0, 1, 2; name : ORN, MC, GC
    subplot(len(spike_monitors), 1, 1 + i)
    plot(spike_monitors[name].t[:]/ (1000 * ms),
         spike_monitors[name].i[:],'|', color='k')
    title(name + ' spike')
    axis('off')

subplots_adjust(wspace=1, hspace=0.5)
show()
#%%
figure(figsize=(16, 9))
num = 1
for M_Sn in M_SN:
    subplot(5,3 , num)
    plot(M_Sn.t[:] /1*second , M_Sn.w[0], color='m', linewidth=2)
    # plt.legend(loc='best')
    plt.xlabel('Time (s)')
    # if num == 1:
    #     title('AN->SON_Synapse3')
    # if num == 2:
    #     title('SON->II_Synapse3')
    # if num == 3:
    #     title('II->IC_Synapse3')
    # if num == 4:
    #     title('AN->IC_Synapse3')
    # if num == 5:
    #     title('AN->IC_Synapse3')
    # if num == 6:
    #     title('IC->THA_Synapse3')
    # if num == 7:
    #     title('THA->AC_CORE_Synapse3')
    # if num == 8:
    #     title('AC_CORE->AC_BELT_Synapse3')
    # if num == 9:
    #     title('AC_BELT->AC_PARA_Synapse3')
    # if num == 10:
    #     title('AC_PARA->OFC_Synapse3')
    # if num == 11:
    #     title('THA->AC_BELT_Synapse3')
    # if num == 12:
    #     title('THA->AC_PARA_Synapse3')
    # if num == 13:
    #     title('AC_BELT->OFC_Synapse3')
    num=num+1
    pass
subplots_adjust(wspace=0.5, hspace=1)
show()
#%%
plt.figure(figsize=(16, 9))
k=1
for ind in connections:
    subplot(5,4,k)
    hist(connections[ind].w,40)
    k+=1
    ylim(0, 10)
    title(ind)

show()
#%%
figure(figsize=(16, 6))

    # i : 0, 1, 2; name : ORN, MC, GC
name='OFC'
plot(spike_monitors[name].t[spike_monitors[name].t[:] / (1000 * ms)>=18] / (1000 * ms),
      spike_monitors[name].i[spike_monitors[name].t[:] / (1000 * ms)>=18] , '|', color='k')
title(name + ' spike')


show()
#%% 可视化突触
from brian2 import *


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(np.array(S.i), np.array(S.j), 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    show()

# start_scope()


G = NeuronGroup(36, 'v:1')
P = NeuronGroup(180, 'v:1')
S = Synapses(G, P)
S.connect('abs(i-j)>=(5*i) and abs(i-j)<=(5*(i+1))')
visualise_connectivity(S)
#%%
figure(figsize=(16, 6))

    # i : 0, 1, 2; name : ORN, MC, GC
name='M8'
# state_monitors['M8']
plot(state_monitors[name].t / (1000 * ms),
      state_monitors[name].v.T/mV)
title(name + ' spike')


show()
#%%
from brian2 import *
# ###########################################
# Defining network model parameters
# ###########################################

NE = 8000          # Number of excitatory cells
NI = NE/4          # Number of inhibitory cells

tau_ampa = 5.0*ms   # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms  # GABAergic synaptic time constant
epsilon = 0.02      # Sparseness of synaptic connections

tau_stdp = 20*ms    # STDP time constant

simtime = 10*second # Simulation time

# ###########################################
# Neuron model
# ###########################################

gl = 10.0*nsiemens   # Leak conductance
el = -60*mV          # Resting potential
er = -80*mV          # Inhibitory reversal potential
vt = -50.*mV         # Spiking threshold
memc = 200.0*pfarad  # Membrane capacitance
bgcurrent = 200*pA   # External current

eqs_neurons='''
dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
dg_ampa/dt = -g_ampa/tau_ampa : siemens
dg_gaba/dt = -g_gaba/tau_gaba : siemens
'''

# ###########################################
# Initialize neuron group
# ###########################################

neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                      reset='v=el', refractory=5*ms, method='euler')
Pe = neurons[:NE]
Pi = neurons[NE:]

# ###########################################
# Connecting the network
# ###########################################

con_e = Synapses(Pe, neurons, on_pre='g_ampa += 0.3*nS')
con_e.connect(p=epsilon)
con_ii = Synapses(Pi, Pi, on_pre='g_gaba += 3*nS')
con_ii.connect(p=epsilon)

# ###########################################
# Inhibitory Plasticity
# ###########################################

eqs_stdp_inhib = '''
w : 1
dApre/dt=-Apre/tau_stdp : 1 (event-driven)
dApost/dt=-Apost/tau_stdp : 1 (event-driven)
'''
alpha = 3*Hz*tau_stdp*2  # Target rate parameter
gmax = 100               # Maximum inhibitory weight

con_ie = Synapses(Pi, Pe, model=eqs_stdp_inhib,
                  on_pre='''Apre += 1.
                         w = clip(w+(Apost-alpha)*eta, 0, gmax)
                         g_gaba += w*nS''',
                  on_post='''Apost += 1.
                          w = clip(w+Apre*eta, 0, gmax)
                       ''')
con_ie.connect(p=epsilon)
con_ie.w = 1e-10

# ###########################################
# Setting up monitors
# ###########################################

sm = SpikeMonitor(Pe)

# ###########################################
# Run without plasticity
# ###########################################
eta = 0          # Learning rate
run(1*second)

# ###########################################
# Run with plasticity
# ###########################################
eta = 1e-2          # Learning rate
run(simtime-1*second, report='text')

# ###########################################
# Make plots
# ###########################################
#%%
figure()
i, t = sm.it
subplot(211)
plot(t/ms, i, 'k.', ms=0.25)
title("Before")
xlabel("")
yticks([])
xlim(0.8*1e3, 1*1e3)
subplot(212)
plot(t/ms, i, 'k.', ms=0.25)
xlabel("time (ms)")
yticks([])
title("After")
xlim((simtime-0.2*second)/ms, simtime/ms)
show()