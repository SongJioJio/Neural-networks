import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F



lr = 0.002 #学习率
momentum = 0.9 #动量超参数
log_interval = 100 #跑多少次batch进行一次日志记录
epochs = 12
train_batch_size = 64
test_batch_size = 1000
img_size = 28


def get_dataloader(train=True):
    assert isinstance(train, bool)

    dataset = torchvision.datasets.MNIST(root='./data', train=train, download=False,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1037,), (0.3081,))
                                         ]))

    batch_size = train_batch_size if train else test_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Sequential( #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),#二维卷积层，in_channels，out_channels，kernel_size，stride，padding=2保证输入输出尺寸相同
            nn.ReLU(), #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2) #output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(), #input_size=(16*10*10)
            nn.MaxPool2d(2, 2) #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #nn.Linear()的输入输出都是维度为1的值
        #所以要把多维度的tensor展平为1维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)  #便于后面交叉熵损失计算


mnist_net = MnistNet() #实例化模型
#optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)
#if os.path.exists("model/mnist_net.pkl"):
    #mnist_net.load_state_dict(torch.load("model/mnist_net.pkl"))
    #optimizer.load_state_dict(torch.load("model/mnist_optimizer.pkl"))
train_loss_list = []
train_count_list = []
accuracy_list = []


def train(epoch):
    #####实现训练过程
    mode = True
    mnist_net.train(mode=mode) #模型设置为训练模式
    train_dataloader = get_dataloader(train=mode) #获取训练数据
    print(len(train_dataloader.dataset))
    print(len(train_dataloader))

    for idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad() #梯度设置为0
        output = mnist_net(data) #调用模型，进行向前计算，获取预测值
        loss = F.nll_loss(output, target) #带权损失函数，真实值为权重，得到损失(tensor)
        loss.backward() #进行反向传播，计算梯度
        optimizer.step() #结束一次前传+反传后，更新梯度

        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_dataloader.dataset),
                100. * idx / len(train_dataloader), loss.item()
            ))
            train_loss_list.append(loss.item())
            train_count_list.append(idx*train_batch_size+(epoch-1)*len(train_dataloader))
            torch.save(mnist_net.state_dict(), "model/mnist_net.pkl") #保存模型参数
            torch.save(optimizer.state_dict(), "model/mnist_optimizer.pkl") #保存优化器参数


def test():
    test_loss = 0
    correct = 0
    mnist_net.eval() #设置模型为评估模式
    test_dataloader = get_dataloader(train=False) #获取测试数据集
    with torch.no_grad(): #不计算其梯度
        for data, target in test_dataloader:
            output = mnist_net(data) #[batch_size, 10]
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] #获取最大值的位置，output[batch_size, 10], target[batch_size, 1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset) #计算平均损失
    print('\nTest set:AVG.loss:{:.4f}, Accuracy:{}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)
    ))
    accuracy_list.append(100.*correct/len(test_dataloader.dataset))


if __name__ == '__main__':
    optimizer = optim.SGD(mnist_net.parameters(), lr=lr, momentum=momentum)
    for epoch in range(1, epochs+1):
        train(epoch)
        test()






