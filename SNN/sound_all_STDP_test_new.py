# %%
import  os
from brian2 import *
import brian2 as b2
import numpy as np
defaultclock.dt = 1*ms
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


def get_matrix_from_file(fileName,I_size,J_size):
    readout = np.load(fileName)
    print('see readout: ', readout.shape, fileName)
    # readout : [..., [neuron i, neuron j, w], ...]
    # value_array.shape = (sensor_number x sensor_number)
    value_array = np.zeros((I_size, J_size))
    if not readout.shape == (0,):
        # value_array[1, 1] = w1, value_array[2, 2] = w2, ......
        value_array[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]
    print(value_array)
    return value_array


txt_folder_path = '/mnt/data/CCM/snndatabase/snncodingdata_18N_5s/'
# txt_folder_path = '/mnt/data/CCM/snndatabase/RWCP_DATA/RWCP_train/'
txt_file_path = findfile(txt_folder_path, '.txt')
print(txt_file_path)

norm_data_list = []
for txt_file in txt_file_path:
    data_per_file = [[] for i in range(18)]
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

# shuffle_indices = np.arange(norm_data_list.shape[0])
# np.random.shuffle(shuffle_indices)
# norm_data_shuffle_list = norm_data_list[shuffle_indices]

sensor_indices_copy_list = []
input_spike_times_list = []
for i in range(len(norm_data_list)):
    scaling_factor_time = 1 * ms
    spikes_time_matrix = norm_data_list[i]
    sensor_indices = [[] for i in range(18)]
    spike_times_list = [[] for i in range(18)]

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
# %%
# sensor_numbers = 5
# run_time = 4000 * ms
sensor_numbers = 18
sensor_numbers_V=100
run_time = 6000 * ms
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
input_groups['input_neurons'] = SpikeGeneratorGroup(sensor_numbers, indices=[0], times=[0 * ms])

# ORN
neuron_groups['AN'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
neuron_groups['AN'].tau = 50 * ms
# neuron_groups['AN'].tau = 50 * ms

neuron_groups['SON'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['SON'].tau = 800 * ms
neuron_groups['SON'].tau = 50 * ms

neuron_groups['II'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['II'].tau = 800 * ms
neuron_groups['II'].tau = 50 * ms

neuron_groups['IC'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['IC'].tau = 800 * ms
neuron_groups['IC'].tau = 50 * ms

neuron_groups['THA'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['THA'].tau = 800 * ms
neuron_groups['THA'].tau = 50 * ms

neuron_groups['AC_CORE'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['AC_CORE'].tau = 800 * ms
neuron_groups['AC_CORE'].tau = 300 * ms

neuron_groups['AC_BELT'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['AC_BELT'].tau = 800 * ms
neuron_groups['AC_BELT'].tau = 300 * ms

neuron_groups['AC_PARA'] = NeuronGroup(sensor_numbers_V, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['AC_PARA'].tau = 800 * ms
neuron_groups['AC_PARA'].tau = 300 * ms

neuron_groups['OFC'] = NeuronGroup(10, eqs1, threshold='v>1', reset='v = 0', method='euler')
# neuron_groups['OFC'].tau = 800 * ms
neuron_groups['OFC'].tau = 300 * ms


weight_path = '/mnt/data/CCM/snndatabase/weights_18N_5s_backup/'
# weight_path = '/mnt/data/CCM/snndatabase/RWCP_DATA/weights/'
starting = 'sound_'
ending = '_1'

# define STDP connections
taupre = 20 * ms
taupost = 20 * ms
wmax = 1
Apre = 0.01
Apost = -Apre * taupre / taupost * 2.05
eqs_stdp = '''
             w : 1
             dapre/dt = -apre/taupre : 1 (clock-driven)
             dapost/dt = -apost/taupost : 1 (clock-driven)
           '''
eqs_stdp_pre = '''
             v_post += w
             apre += Apre*(1-w)
             w = clip(w+apost, 0, wmax)
             '''
eqs_stdp_post = '''
             apost += Apost*w
             w = clip(w+apre, 0, wmax)
             '''


# input -> ORN
connections['SI'] = Synapses(input_groups['input_neurons'], neuron_groups['AN'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
# connections['SI'].connect(j='i')
connections['SI'].connect('abs(i-j)>=(5*i) and abs(i-j)<=(5*(i+1))')
weightMatrix_S0 = get_matrix_from_file(weight_path + starting + 'SI' + ending + '.npy',sensor_numbers,sensor_numbers_V)
connections['SI'].w = weightMatrix_S0[connections['SI'].i, connections['SI'].j]


# ORN -> MC
connections['S0'] = Synapses(neuron_groups['AN'], neuron_groups['SON'], model='''w : 1''',
                                  on_pre='''v += w''', method='exact')
# connections['S0'].connect(p=1)
connections['S0'].connect(j='i')
weightMatrix_S1 = get_matrix_from_file(weight_path + starting + 'S0' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S0'].w = weightMatrix_S1[connections['S0'].i, connections['S0'].j]

connections['S1'] = Synapses(neuron_groups['SON'], neuron_groups['II'],model='''w : 1''',
                                      on_pre='''v += w''', method='exact')
connections['S1'].connect(j='i')
weightMatrix_S2 = get_matrix_from_file(weight_path + starting + 'S1' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S1'].w = weightMatrix_S2[connections['S1'].i, connections['S1'].j]

connections['S2'] = Synapses(neuron_groups['II'], neuron_groups['IC'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S2'].connect(j='i')
weightMatrix_S3 = get_matrix_from_file(weight_path + starting + 'S2' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S2'].w = weightMatrix_S3[connections['S2'].i, connections['S2'].j]


connections['S3'] = Synapses(neuron_groups['AN'], neuron_groups['IC'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S3'].connect(j='i')
weightMatrix_S4 = get_matrix_from_file(weight_path + starting + 'S3' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S3'].w = weightMatrix_S4[connections['S3'].i, connections['S3'].j]

connections['S4'] = Synapses(neuron_groups['SON'], neuron_groups['IC'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S4'].connect(j='i')
weightMatrix_S5 = get_matrix_from_file(weight_path + starting + 'S4' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S4'].w = weightMatrix_S5[connections['S4'].i, connections['S4'].j]

connections['S5'] = Synapses(neuron_groups['IC'], neuron_groups['THA'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S5'].connect(j='i')
weightMatrix_S6 = get_matrix_from_file(weight_path + starting + 'S5' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S5'].w = weightMatrix_S6[connections['S5'].i, connections['S5'].j]

connections['S6'] = Synapses(neuron_groups['THA'], neuron_groups['AC_CORE'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S6'].connect(j='i')
weightMatrix_S7 = get_matrix_from_file(weight_path + starting + 'S6' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S6'].w = weightMatrix_S7[connections['S6'].i, connections['S6'].j]

connections['S7'] = Synapses(neuron_groups['AC_CORE'], neuron_groups['AC_BELT'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S7'].connect(p=1)
weightMatrix_S8 = get_matrix_from_file(weight_path + starting + 'S7' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S7'].w = weightMatrix_S8[connections['S7'].i, connections['S7'].j]

connections['S8'] = Synapses(neuron_groups['AC_BELT'], neuron_groups['AC_PARA'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S8'].connect(p=1)
weightMatrix_S9 = get_matrix_from_file(weight_path + starting + 'S8' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S8'].w = weightMatrix_S9[connections['S8'].i, connections['S8'].j]

connections['S9'] = Synapses(neuron_groups['AC_BELT'], neuron_groups['OFC'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S9'].connect(p=1)
weightMatrix_S10 = get_matrix_from_file(weight_path + starting + 'S9' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S9'].w = weightMatrix_S10[connections['S9'].i, connections['S9'].j]

connections['S10'] = Synapses(neuron_groups['THA'], neuron_groups['AC_BELT'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S10'].connect(j='i')
weightMatrix_S11 = get_matrix_from_file(weight_path + starting + 'S10' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S10'].w = weightMatrix_S11[connections['S10'].i, connections['S10'].j]

connections['S11'] = Synapses(neuron_groups['THA'], neuron_groups['AC_PARA'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S11'].connect(j='i')
weightMatrix_S12 = get_matrix_from_file(weight_path + starting + 'S11' + ending + '.npy',sensor_numbers_V,sensor_numbers_V)
connections['S11'].w = weightMatrix_S12[connections['S11'].i, connections['S11'].j]

connections['S12'] = Synapses(neuron_groups['AC_PARA'], neuron_groups['OFC'], model='''w : 1''',
                             on_pre='''v += w''', method='exact')
connections['S12'].connect(p=1)
weightMatrix_S13 = get_matrix_from_file(weight_path + starting + 'S12' + ending + '.npy',sensor_numbers_V,10)
connections['S12'].w = weightMatrix_S13[connections['S12'].i, connections['S12'].j]
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

# state_monitors['M8'] = StateMonitor(neuron_groups['OFC'], 'v', record=True)
spike_monitors['OFC'] = SpikeMonitor(neuron_groups['OFC'])


# state_monitors['M_SI'] = StateMonitor(connections['SI'], ['w', 'apre', 'apost'], record=True)

# state_monitors['M_S0'] = StateMonitor(connections['S0'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S1'] = StateMonitor(connections['S1'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S2'] = StateMonitor(connections['S2'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S3'] = StateMonitor(connections['S3'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S4'] = StateMonitor(connections['S4'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S5'] = StateMonitor(connections['S5'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S6'] = StateMonitor(connections['S6'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S7'] = StateMonitor(connections['S7'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S8'] = StateMonitor(connections['S8'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S9'] = StateMonitor(connections['S9'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S10'] = StateMonitor(connections['S10'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S11'] = StateMonitor(connections['S11'], ['w', 'apre', 'apost'], record=True)
# state_monitors['M_S12'] = StateMonitor(connections['S12'], ['w', 'apre', 'apost'], record=True)

# spike_monitors['OFC0'] = SpikeMonitor(neuron_groups['OFC'][0])
# spike_monitors['OFC1'] = SpikeMonitor(neuron_groups['OFC'][1])
# spike_monitors['OFC2'] = SpikeMonitor(neuron_groups['OFC'][2])
# spike_monitors['OFC3'] = SpikeMonitor(neuron_groups['OFC'][3])
# spike_monitors['OFC4'] = SpikeMonitor(neuron_groups['OFC'][4])
# spike_monitors['OFC5'] = SpikeMonitor(neuron_groups['OFC'][5])
# spike_monitors['OFC6'] = SpikeMonitor(neuron_groups['OFC'][6])
# spike_monitors['OFC7'] = SpikeMonitor(neuron_groups['OFC'][7])
# spike_monitors['OFC8'] = SpikeMonitor(neuron_groups['OFC'][8])
# spike_monitors['OFC9'] = SpikeMonitor(neuron_groups['OFC'][9])
# spike_monitors['OFC'] = SpikeMonitor(neuron_groups['OFC'])

# M_SN = [state_monitors['M_S0'], state_monitors['M_S1'],state_monitors['M_S2'], state_monitors['M_S3'],
#         state_monitors['M_S4'], state_monitors['M_S5'],state_monitors['M_S6'], state_monitors['M_S7'],
#         state_monitors['M_S8'], state_monitors['M_S9'],state_monitors['M_S10'], state_monitors['M_S11'],state_monitors['M_S12']]
#%%
net = Network()
for obj_list in [input_groups, neuron_groups, connections, state_monitors, spike_monitors]:
    for key in obj_list:
        net.add(obj_list[key])

# 先让net跑0秒是为了消除spikegeneratorgroup在0秒处的脉冲影响
# net.run(0*second)


num_examples = 0
MC_spike_monitor_temp = [[] for i in range(10)]
for neuron_indices, neuron_spike_times in zip(sensor_indices_copy_list, input_spike_times_list):
    input_groups['input_neurons'].set_spikes(indices=neuron_indices, times=neuron_spike_times + num_examples * run_time)
    net.run(run_time)
    new_txt_file = open('/mnt/data/CCM/snndatabase/aftercoding_18N_5s/%s.txt' %
    # new_txt_file = open('/mnt/data/CCM/snndatabase/RWCP_DATA/RWCP_aftercoding_train_fromAuro/%s.txt' %
                        (txt_file_path[num_examples].split('/')[-2].split('.t')[0]+'-'+txt_file_path[num_examples].split('/')[-1].split('.t')[0]), 'w', encoding='utf-8')
    # MC_spike_monitors = [spike_monitors['OFC0'], spike_monitors['OFC1'],spike_monitors['OFC2'],spike_monitors['OFC3'],spike_monitors['OFC4'],spike_monitors['OFC5'],spike_monitors['OFC6'],spike_monitors['OFC7'],spike_monitors['OFC8'],spike_monitors['OFC9']]
    # for num_MC in range(len(MC_spike_monitors)):
    #     # 求出一个样本的脉冲发放时刻：net处理完该样本后新增的脉冲发放时刻
    #     delta_MC_spike_times = [x for x in MC_spike_monitors[num_MC].t[:] if x not in MC_spike_monitor_temp[num_MC]]
    #     delta_MC_spike_times -= num_examples * run_time
    #     print('MC spike times: ', delta_MC_spike_times)
    #     for MC_spike_time in delta_MC_spike_times:
    #         new_txt_file.writelines(str('%.4f' % (MC_spike_time/(1000*ms))) + ' ')
    #     new_txt_file.writelines('\n')
    #     MC_spike_monitor_temp[num_MC] = MC_spike_monitors[num_MC].t[:]

    for num_MC in range(10):
        # 求出一个样本的脉冲发放时刻：net处理完该样本后新增的脉冲发放时刻
        delta_MC_spike_times = [x for x in spike_monitors['OFC'].t[spike_monitors['OFC'].i==num_MC] if x not in MC_spike_monitor_temp[num_MC]]
        delta_MC_spike_times -= num_examples * run_time
        print('MC spike times: ', delta_MC_spike_times)
        for MC_spike_time in delta_MC_spike_times:
            new_txt_file.writelines(str('%.4f' % (MC_spike_time/(1000*ms))) + ' ')
        new_txt_file.writelines('\n')
        MC_spike_monitor_temp[num_MC] = spike_monitors['OFC'].t[spike_monitors['OFC'].i==num_MC]
    new_txt_file.close()
    print("----------the network has tested %s samples----------" % (num_examples + 1))
    num_examples += 1

net.stop()
#%%
# from brian2 import *
#
# def get_matrix_from_file(fileName):
#     readout = np.load(fileName)
#     print('see readout: ', readout.shape, fileName)
#     # readout : [..., [neuron i, neuron j, w], ...]
#     # value_array.shape = (sensor_number x sensor_number)
#     value_array = np.zeros((250, 250))
#     if not readout.shape == (0,):
#         # value_array[1, 1] = w1, value_array[2, 2] = w2, ......
#         value_array[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]
#     print(value_array)
#     return value_array
# weight_path = '/mnt/data/CCM/snndatabase/weights_new_2/'
# starting = 'sound_'
# ending = '_1'
# weightMatrix_S1 = get_matrix_from_file(weight_path + starting + 'S0' + ending + '.npy')