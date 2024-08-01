import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 声明参数
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 创建虚拟数据集类
class RandomDataset(Dataset):  # 继承torch.utils.data.Dataset抽象父类，要实现getitem抽象方法

    def __init__(self, size, length):  # RandomDataset类的构造器
        self.len = length
        self.data = torch.randn(length, size)  # 创建一个length×size大小的张量

    def __getitem__(self, index):  # 实现父类中的抽象方法
        return self.data[index]

    def __len__(self):
        return self.len


# 实例化数据集加载器
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True, num_workers=12)


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size,
              "output size", output.size)
        return output


model = Model(input_size, output_size)  # 实例化模型对象
if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型

model.to(device)  # 把并行的模型移动到GPU上


for data in rand_loader:
    input = data.to(device)  # 把输入张量移到GPU
    output = model(input)
    print(f"Outside: input size {input.size()},"
          f"output_size {output.size()}")
