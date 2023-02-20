import torch
import time
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
import torch.linalg as linalg
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

"""
use contraction regularized NODE to learn a dynamics to map two initial points to two terminal points.
The distance between the two intial points is smaller than the distance between the two terminal points.
the code is used to check whether contraction regularized NODE works.
"""


global_train_contraction_metric_flag = True
global_max_epoches = 500
global_train_flag = True
global_data_size = 2
global_optimizer_lr = 5e-2
global_weight_decay = 1e-2


class TrainDataset(Dataset):
    def __init__(self):
        self.InitialPoints = torch.zeros(2, 2)
        self.TerminalPoints = torch.zeros(2, 2)

        self.InitialPoints[0, :] = torch.tensor([2, 2])
        self.InitialPoints[1, :] = torch.tensor([2, -2])

        self.TerminalPoints[0, :] = torch.tensor([4, 4])
        self.TerminalPoints[1, :] = torch.tensor([4, -4])

    def __len__(self):
        return len(self.InitialPoints)

    def __getitem__(self, idx):
        return self.InitialPoints[idx, :], self.TerminalPoints[idx, :]


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 6),
            nn.Tanh(),
            nn.Linear(6, 2),
        )

        #权重初始化
        #if global_train_contraction_metric_flag:
            #my_init_weight0 = torch.normal(0, 0.01, size=(6, 2), requires_grad=True)
            #self.net[0].weight = nn.Parameter(my_init_weight0)
            #my_init_weight1 = torch.normal(0, 0.01, size=(6, 6), requires_grad=True)
            #self.net[2].weight = nn.Parameter(my_init_weight1)
            #my_init_weight2 = torch.normal(0, 0.01, size=(2, 6), requires_grad=True)
            #self.net[4].weight = nn.Parameter(my_init_weight2)

    def forward(self, t, x):
        return self.net(x)


class regression_node(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.t = torch.linspace(0., 0.5, 50)
        self.func = ODEFunc()

    def node_propagation(self, x0):
        # traj_x = odeint(self.func, x0, t, method='dopri5')
        traj_x = odeint(self.func, x0, self.t, method='euler')
        return traj_x

    def forward(self, x0):
        return self.node_propagation(x0)[-1]

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(
            self.func.parameters(), lr=global_optimizer_lr)

        def lambda1(epoch): return 0.99 ** epoch

        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            optimizer1, lr_lambda=lambda1)

        return [optimizer1], [scheduler1]

    def training_step(self, train_batch):

        lossFunc = nn.MSELoss()

        x0, y = train_batch
        loss = lossFunc(self.forward(x0), y)

        if global_train_contraction_metric_flag:
            alpha = 1
            #对角阵元素范围[a, b]
            a = 0.1
            b = 1
            l1 = torch.zeros((6, 2), requires_grad=True)
            u1 = torch.zeros((6, 2), requires_grad=True)
            l2 = torch.zeros((6, 2), requires_grad=True)
            u2 = torch.zeros((6, 2), requires_grad=True)
            l3 = torch.zeros((6, 2), requires_grad=True)
            u3 = torch.zeros((6, 2), requires_grad=True)
            l4 = torch.zeros((2, 2), requires_grad=True)
            u4 = torch.zeros((2, 2), requires_grad=True)
            contraction_regularizer = torch.zeros((1, 1), requires_grad=True)

            weight0 = self.func.net[0].weight
            weight1 = self.func.net[2].weight
            weight2 = self.func.net[4].weight

            #STEP1
            for i in range(weight0.size()[0]):
                for j in range(weight0.size()[1]):
                    if weight0[i, j] > 0:
                        with torch.no_grad():
                            l1[i, j] = a * weight0[i, j]
                            u1[i, j] = b * weight0[i, j]
                    else:
                        with torch.no_grad():
                            l1[i, j] = b * weight0[i, j]
                            u1[i, j] = a * weight0[i, j]

            #STEP2
            for i in range(weight1.size()[0]):
                for j in range(weight0.size()[1]):
                    for k in range(weight1.size()[1]):
                        if weight1[i, k] > 0:
                            with torch.no_grad():
                                l2[i, j] += weight1[i, k] * l1[k, j]
                                u2[i, j] += weight1[i, k] * u1[k, j]
                        else:
                            with torch.no_grad():
                                l2[i, j] += weight1[i, k] * u1[k, j]
                                u2[i, j] += weight1[i, k] * l1[k, j]

                    #STEP3
                    if l2[i, j] < 0:
                        if u2[i, j] < 0:
                            with torch.no_grad():
                                l3[i, j] = b * l2[i, j]
                                u3[i, j] = a * l2[i, j]
                        else:
                            with torch.no_grad():
                                l3[i, j] = b * u2[i, j]
                                u3[i, j] = b * l2[i, j]
                    else:
                        with torch.no_grad():
                            l3[i, j] = a * l2[i, j]
                            u3[i, j] = b * l2[i, j]

            #STEP4
            for i in range(weight2.size()[0]):
                for j in range(weight0.size()[1]):
                    for k in range(weight2.size()[1]):
                        if weight2[i, k] > 0:
                            with torch.no_grad():
                                l4[i, j] += weight2[i, k] * l3[k, j]
                                u4[i, j] += weight2[i, k] * u3[k, j]
                        else:
                            with torch.no_grad():
                                l4[i, j] += weight2[i, k] * u3[k, j]
                                u4[i, j] += weight2[i, k] * l3[k, j]

            Nc = weight0.size()[1]
            for i in range(Nc):
                with torch.no_grad():
                    contraction_regularizer += F.relu(-abs(alpha + 2 * (l4[i, i]+u4[i, i])) +
                                                      torch.sum(torch.abs(u4[i, :])) +
                                                      torch.sum(torch.abs(u4[:, i])))

            return 10*loss + contraction_regularizer
        else:
            return loss

    def test_step(self, batch, batch_idx):

        data, label = batch

        # plot points around the train point and the propatation
        number_of_samples = 50
        radius = 0.5 #半径
        angle = 2*3.14*torch.linspace(0, 1, number_of_samples)
        for i in range(0, len(data), 1):
            plt.plot(data[i, 0], data[i, 1], 'b.')
            plt.plot(label[i, 0], label[i, 1], 'r*')

            sample = data[i]+torch.stack([radius*torch.cos(angle),
                                          radius*torch.sin(angle)]).T
            plt.plot(sample[:, 0], sample[:, 1], 'b')
            traj = self.node_propagation(sample)
            plt.plot(traj[-1, :, 0], traj[-1, :, 1], 'r')

        # plot the trajectory

        data_propagation = self.node_propagation(data)
        for i in range(len(data)):
            ith_traj = data_propagation[:, i, :]

            plt.plot(ith_traj[:, 0].cpu(),
                     ith_traj[:, 1].cpu(), 'g')

        xv, yv = torch.meshgrid(torch.linspace(0, 5, 30),
                                torch.linspace(-5, 5, 30))

        y1 = torch.stack([xv.flatten(), yv.flatten()])

        vector_field = self.func.net(y1.T)
        u = vector_field[:, 0].reshape(xv.size())
        v = vector_field[:, 1].reshape(xv.size())

        plt.quiver(xv, yv, u, v)

        plt.savefig('node_propatation.pdf')

        return 1


# data


if __name__ == '__main__':

    plt.figure(1)
    training_data = TrainDataset()
    train_dataloader = DataLoader(
        training_data, batch_size=global_data_size, num_workers=4)

    model = regression_node()

    trainer = pl.Trainer(gpus=None,  num_nodes=1,
                         max_epochs=global_max_epoches)

    if global_train_flag:
        trainer.fit(model, train_dataloader)
        trainer.save_checkpoint("example_2d.ckpt")

    time.sleep(5)

    new_model = regression_node.load_from_checkpoint(
        checkpoint_path="example_2d.ckpt")

    plt.figure(1)
    trainer = pl.Trainer(gpus=None,  num_nodes=1,
                         max_epochs=global_max_epoches)
    success_rate = trainer.test(new_model, train_dataloader)
    print("after training, the successful predition rate on train set is", success_rate)
    plt.show()
    plt.close(fig=1)

    fig = plt.figure(2)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([2, 3, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.grid(which='both')

    ax.plot(2, 2, 'b.')
    ax.plot(2, -2, 'b.')
    ax.plot(4, 4, 'r*')
    ax.plot(4, -4, 'r*')
    ax.text(2.05, 2.2, r"$x_0^1$", fontsize=16)
    ax.text(2.05, -1.8, r"$x_0^2$", fontsize=16)
    ax.text(3.85, 3.5, r"$x_T^1$", fontsize=16)
    ax.text(3.85, -3.8, r"$x_T^2$", fontsize=16)
    plt.savefig('2d_learn_task.pdf')
    plt.close(fig=2)

