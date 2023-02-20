import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl


lr = 0.002 #学习率
momentum = 0.9 #动量超参数
train_batch_size = 64
test_batch_size = 1000


class MyDataModule(pl.LightningDataModule):
    def __int__(self):
        super().__init__()
        self.dim = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        torchvision.datasets.MNIST(root='./data', train=True, download=False)
        torchvision.datasets.MNIST(root='./data', train=False, download=False)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.mnist_train = torchvision.datasets.MNIST(root='./data', train=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1037,), (0.3081,))
                                                    ]))

        if stage == 'test' or stage is None:
             self.mnist_test = torchvision.datasets.MNIST(root='./data', train=False,
                                                          transform=torchvision.transforms.Compose([
                                                              torchvision.transforms.ToTensor(),
                                                              torchvision.transforms.Normalize((0.1037,), (0.3081,))
                                                          ]))

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=train_batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=test_batch_size, num_workers=4)


class MnistNet(pl.LightningModule):
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

    def configure_optimizers(self):
        optimizer = optim.SGD(mnist_net.parameters(), lr=lr, momentum=momentum)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = mnist_net(x) #调用模型，进行向前计算，获取预测值
        loss = F.nll_loss(output, y) #带权损失函数，真实值为权重，得到损失(tensor)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        test_loss = 0
        correct = 0
        x, y = batch
        output = mnist_net(x)  # [batch_size, 10]
        test_loss += F.nll_loss(output, y, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置，output[batch_size, 10], target[batch_size, 1]
        correct += pred.eq(y.data.view_as(pred)).sum()
        test_loss /= 10000  # 计算平均损失
        self.log('test_loss', test_loss)

    #def test_epoch_end(self, train_step_outputs):
        #for out in train_step_outputs:


#def mnist():
    #dataset = MyDataModule()
    #trainer = pl.Trainer(max_epochs=1)
    #trainer.fit(mnist_net, datamodule=dataset)
    #trainer.test(mnist_net, datamodule=dataset, verbose=True)


if __name__ == '__main__':
    mnist_net = MnistNet()  # 实例化模型
    dataset = MyDataModule()
    trainer = pl.Trainer(max_epochs=12)
    trainer.fit(mnist_net, datamodule=dataset)
    trainer.test(mnist_net, datamodule=dataset, verbose=True)
