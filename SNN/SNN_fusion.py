import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

# from SNN.Network import SNNforauditory
from Network import SNNforauditory
from spikingjelly import visualizing
#%%
# 获取样本地址
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

def read_origindata(dir):#读取初始编码
    # 读取数据
    folder_nums = 0  # 从第0个文件夹开始，遍历到第3个，共4个文件夹
    file_nums_count = 0  # 文件计数器
    data = []  # 总的训练集集合
    labels = []  # 创建每组数据对应的标签
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        print(folder_nums, folder, folder_path)  # 3 3_腐败水果 ./data/txt_data_for_food/spiking_train/3_腐败水果
        file_path = findfile(folder_path, '.txt')
        label = folder.split('_')[0]
        for file in file_path:
            # print(file)
            # 补零数不影响训练结果，矩阵的第二维大于单个神经元的最大脉冲数量即可
            data_per_file = np.zeros((25, 1000))
            f = open(file, 'r')
            content = f.readlines()
            f.close()
            # print(len(content)):6
            row = 0
            for items in content:
                data_i = items.split()
                # print(data_i)
                col = 0
                for x in data_i:
                    data_per_file[row][col] = x
                    col += 1
                row += 1
            # print(data_per_file)
            data.append(data_per_file)
            labels.append(int(label))
            file_nums_count += 1
            # print(file_nums_count, '\n')
        folder_nums += 1
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def read_data(dir):#读取编码后
    data = []  # 总的训练集集合
    labels = []  # 创建每组数据对应的标签
    for folder in os.listdir(dir):
        file_path = os.path.join(dir, folder)
        label=int(folder.split('-')[0])
        data_per_file = np.zeros((10, 2000))
        f = open(file_path, 'r')
        content = f.readlines()
        f.close()
        row = 0
        for items in content:
            data_i = items.split()
            col = 0
            for x in data_i:
                data_per_file[row][col] = x
                col += 1
            row += 1
        data.append(data_per_file)
        labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def timetobinary(timearray, T):
    binscal=5001/T
    x = np.asarray(timearray*1000/ binscal, dtype=int)
    spike_array = np.zeros(int(5001/ binscal), dtype=bool)
    for i in x:
        spike_array[i] = True
    return spike_array

def change_data(T,data):
    # 将数组转为01的spike

    spikes_result = []
    for num_example in range(data.shape[0]):
        example_spike_time_list = []
        tmp = data[num_example]
        for row in range(data.shape[1]):
            example_spike_time_list.append(
                timetobinary(tmp[row], T))
        spikes_result.append(example_spike_time_list)

    spikes_result=np.array(spikes_result)
    print(spikes_result.shape)

    # # print(data_X, labels_Y, data_X.shape, labels_Y.shape)
    return spikes_result

# 将训练集和测试集包装为loader形式
def data_package(train_X, train_Y, test_X, test_Y, batch_size):
    # numpy数据转tensor
    x_train_data = torch.from_numpy(train_X).type(torch.FloatTensor)
    y_train_data = torch.from_numpy(train_Y).type(torch.FloatTensor)
    x_test_data = torch.from_numpy(test_X).type(torch.FloatTensor)
    y_test_data = torch.from_numpy(test_Y).type(torch.FloatTensor)
    print(x_train_data.size(), y_train_data.size(), x_test_data.size(), y_test_data.size())

    # 将数据tensor和标签tensor包装成Dataset类
    train_dataset = TensorDataset(x_train_data, y_train_data)
    test_dataset = TensorDataset(x_test_data, y_test_data)

    # 将dataset传入DataLoader中，shuffle使得每个epoch中的样本生成顺序不一样，也就是不依次取样本
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
#%%
# def main():
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Total GPU numbers:' + str(torch.cuda.device_count()), '\n', 'Being uesd GPU:' + str(device))
# 数据集路径
# train_dir = '/mnt/data/CCM/snndatabase/snncodingdata_36N_5s/'
train_dir='/mnt/data/CCM/snndatabase/fusion2/'
# train_dir = '/mnt/data/CCM/snndatabase/aftercoding_18N_5s/'
# 每批数据样本量，根据显存大小设置
batch_size = 128
# 学习率
learning_rate = 1e-2
# 仿真时长，T 越大，仿真占用显存越大
T =20
# LIF神经元的时间常数tau，tau 越大，拟合速度越慢
tau = 2.0
# 训练轮数，经实验，一般需要设置为 10*tau 以上
train_epoch = 1000
# 保存tensorboard日志文件的位置
log_dir = ''

# 读取数据并处理
train_data, train_labels = read_origindata(train_dir)
# train_labels=np.expand_dims(train_labels,axis=1)
# train_data, train_labels = read_data(train_dir)
(X_train, X_test, Y_train, Y_test) = train_test_split(train_data, train_labels, test_size=0.1, random_state=25)
train_X = change_data(T, X_train)
test_X = change_data(T, X_test)

#包装数据
train_loader, test_loader = data_package(train_X, Y_train,test_X,  Y_test, batch_size)
# # 为特定GPU设置种子，生成随机数，设置随机种子是为了确保每次生成固定的随机数，以使得每次实验结果一致
torch.cuda.manual_seed(0)

# 初始化网络

# net = SNNforauditory(tau=tau)  # 实例化模型对象
# if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
#     # print(f"Let's use {torch.cuda.device_count()} GPUs!")
#     net = nn.DataParallel(net)  # 将模型对象转变为多GPU并行运算的模型 model = nn.DataParallel(model，device_ids=[0,1,2])
#
# net.to(device) # 把并行的模型移动到GPU上

net = SNNforauditory(tau=tau).to(device)

# 使用Adam优化器
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
train_times = 0
max_test_accuracy = -1

train_accuracy_all = []
train_loss_all = []

validation_accuracy_all = []
validation_loss_all = []

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
        # label=np.expand_dims(label,axis=1)
        # label = label.float().to(device)
        label = label.long().to(device)
        # data = data.cuda()
        # label = label.long().cuda()

        label_one_hot = F.one_hot(label, 3).float()
        data_seq = data.permute(2, 0, 1)

        out_spikes_counter_frequency =0

        for t in range(T):
            out_spikes_counter_frequency+=net(data_seq[t])
        out_spikes_counter_frequency=out_spikes_counter_frequency/T


        # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
        # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
        # OUT_FEQ=torch.round(out_spikes_counter_frequency)
        # loss= F.mse_loss(out_spikes_counter_frequency, label)
        loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
        # loss=F.cross_entropy(out_spikes_counter_frequency, label_one_hot)
        loss.backward()
        optimizer.step()
        # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
        functional.reset_net(net)
        train_losses.append(loss.item())

        # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
        train_correct_sum += ( out_spikes_counter_frequency.max(1)[1] == label).float().sum().item()
        # train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label).float().sum().item()
        # train_correct_sum +=(torch.round(out_spikes_counter_frequency) == label).sum().item()
        train_sum += label.numel()
        train_accuracy = train_correct_sum / train_sum

        if train_times % 1 == 0:
            print('train_times:', train_times, 'train_loss:', '%.5f' % (loss.item()), 'train_accuracy:',
                  '%.5f' % train_accuracy)


        epoch_train_accuracy.append(train_accuracy)
    epoch_train_loss = np.array(train_losses).mean()
    train_loss_all.append(epoch_train_loss)
    train_accuracy_all.append(epoch_train_accuracy[-1])

    # 验证模型
    net.eval()
    with torch.no_grad():
        # 每遍历一次全部数据集，就在测试集上测试一次
        test_sum = 0
        correct_sum = 0
        val_losses = []
        for test_data, test_label in test_loader:
            test_data = test_data.to(device)
            # test_label = test_label.float().to(device)
            test_label = test_label.long().to(device)
            test_label_one_hot = F.one_hot(test_label,3).float()
            test_data_seq = test_data.permute(2, 0, 1)

            test_output = 0
            for t in range(T):
                test_output +=net(test_data_seq[t])  # batch_size x 4 tensor
            test_output = test_output / T


            val_loss = F.mse_loss(test_output, test_label_one_hot)
            # val_loss = F.cross_entropy(test_output, test_label_one_hot)
            val_losses.append(val_loss.item())
            correct_sum += (test_output.max(1)[1] == test_label).float().sum().item()  # 预测正确的样本数
            test_sum += test_label.numel()  # numel()用来返回数组中元素的个数
            # print('test sample numbers:', test_sum)
            functional.reset_net(net)

        epoch_val_loss = np.array(val_losses).mean()
        validation_loss_all.append(epoch_val_loss)
        test_accuracy = correct_sum / test_sum

        print('Epoch %s' % (epoch + 1), 'test_loss:', '%.5f' % epoch_val_loss, 'test_accuracy:',
              '%.5f' % test_accuracy)

        validation_accuracy_all.append(test_accuracy)

        if max_test_accuracy < test_accuracy:
            max_test_accuracy = test_accuracy
            # print('saving net...')
            # torch.save(net, './models/ccm_snn_model_01.pth')
            # print('saved model successfully')

    print(
        'device={}, dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
            device, train_dir, batch_size, learning_rate, T, log_dir, max_test_accuracy, train_times))

# 保存loss和acc到txt文件中
# txt_file = open('./figures/ccm_loss_acc_copt_bsa_snn_1.txt', 'w', encoding='utf-8')
# for element in train_accuracy_all:
#     txt_file.writelines(str(element) + ' ')
# txt_file.writelines('\n')
# for element in validation_accuracy_all:
#     txt_file.writelines(str(element) + ' ')
# txt_file.writelines('\n')
# for element in train_loss_all:
#     txt_file.writelines(str(element) + ' ')
# txt_file.writelines('\n')
# for element in validation_loss_all:
#     txt_file.writelines(str(element) + ' ')
# txt_file.writelines('\n')
# txt_file.close()
#%%
# 绘制loss和acc曲线
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
#
#
# #%%
#
#
# if __name__ == '__main__':
#     main()
# #%%
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print('Total GPU numbers:' + str(torch.cuda.device_count()), '\n', 'Being uesd GPU:' + str(device))
#     # 数据集路径
#     train_dir = '/mnt/data/CCM/snndatabase/snncodingdata_36N_5s/'
#     # train_dir = '/mnt/data/CCM/snndatabase/aftercoding_18N_5s/'
# #     #
#     # 每批数据样本量，根据显存大小设置
#     batch_size = 64
#     # 学习率
#     learning_rate = 1e-2
#     # 仿真时长，T 越大，仿真占用显存越大
#
#     # LIF神经元的时间常数tau，tau 越大，拟合速度越慢
#     tau = 2.0
#     # 训练轮数，经实验，一般需要设置为 10*tau 以上
#     train_epoch = 300
#     # 保存tensorboard日志文件的位置
#     log_dir = ''
#     #
#     # 读取数据并处理
#     train_data, train_labels = read_origindata(train_dir)
#     # train_data, train_labels = read_data(train_dir)
#     (X_train, X_test, Y_train, Y_test) = train_test_split(train_data, train_labels, test_size=0.25, random_state=25)
#
#     #
#     T = 200
#     train_X = change_data(T, X_train)
#     test_X = change_data(T, X_test)
#     #
#     #包装数据
#     train_loader, test_loader = data_package(train_X, Y_train,test_X,  Y_test, batch_size)
#
#
#     from spikingjelly import visualizing
#     for ind in range(10):
#         plt.figure()
#         visualizing.plot_1d_spikes(train_X[ind,:,:].T,'PeriodicEncoder', 'Simulating Step', 'Neuron Index',plot_firing_rate=False)
#     # visualizing.plot_1d_spikes(train_X[1,:,:].T,'PeriodicEncoder', 'Simulating Step', 'Neuron Index',plot_firing_rate=False)
#     # # visualizing.plot_1d_spikes(train_X[2,:,:].T,'PeriodicEncoder', 'Simulating Step', 'Neuron Index',plot_firing_rate=False)
#     # # visualizing.plot_1d_spikes(train_X[3,:,:].T,'PeriodicEncoder', 'Simulating Step', 'Neuron Index',plot_firing_rate=False)
#         print(Y_train[ind])
#         plt.show()

