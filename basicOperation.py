import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# 假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

# 1 快速搭建网络
# net2 = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 1)
# )

# print(net2)



# # Part2 保存网络
#
# def save():
#     # 建网络
#     net1 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )
#     optimizer = torch.optim.SGD(net1.parameters(), lr=0.2) # the orginal is lr=0.5, but the performance is bad, the result will stop change soon.
#     loss_func = torch.nn.MSELoss()
#
#     # 训练
#     for t in range(100):
#         prediction = net1(x)
#         loss = loss_func(prediction, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print("\n prediction \n", prediction)
#         if(t%5==0):
#             print("\n prediction \n",prediction)
#     # Method1
#     torch.save(net1, 'net.pkl')  # 保存整个网络
#     # Method2
#     torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
#
# save()


# # Part3 提取网络
# print(x)
# ## Method1 提取整个网络，网络较大时会比较慢
# def restore_net():
#     # restore entire net1 to net2
#     net2 = torch.load('net.pkl')
#     prediction = net2(x)
#     # print(prediction)
# ## Method2 只提取网络参数，提取所有参数，然后再放到你的新建网络中；
# def restore_params():
#     # 新建 net3
#     net3 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )
#
#     # 将保存的参数复制到 net3
#     net3.load_state_dict(torch.load('net_params.pkl'))
#     prediction = net3(x)
#     # print(prediction)
#
# restore_net()
# restore_params()

# 批训练
import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5      # 批训练的数据个数

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# print("\nx ",x)
# print("\ny ",y)


# 先转换成 torch 能识别的 Dataset
# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
torch_dataset = Data.TensorDataset(x, y)
print(torch_dataset)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

for epoch in range(3):   # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...

        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())


# #优化器效果对比
# import torch
# import torch.utils.data as Data
# import torch.nn.functional as F
# import matplotlib.pyplot as plt


# # 伪数据
# torch.manual_seed(1)    # reproducible

# LR = 0.01
# BATCH_SIZE = 32
# EPOCH = 12

# # fake dataset
# x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# # # plot dataset
# # plt.scatter(x.numpy(), y.numpy())
# # plt.show()

# # 使用上节内容提到的 data loader
# torch_dataset = Data.TensorDataset(x, y)
# loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)

# # 每个优化器优化
# # 默认的 network 形式
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(1, 20)   # hidden layer
#         self.predict = torch.nn.Linear(20, 1)   # output layer

#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.predict(x)             # linear output
#         return x

# # 为每个优化器创建一个 net
# net_SGD         = Net()
# net_Momentum    = Net()
# net_RMSprop     = Net()
# net_Adam        = Net()
# nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# # different optimizers
# opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
# opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
# opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
# opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
# optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

# loss_func = torch.nn.MSELoss()
# losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss

# # Training & Visualization 
# for epoch in range(EPOCH):
#     print('Epoch: ', epoch)
#     for step, (b_x, b_y) in enumerate(loader):

#         # 对每个优化器, 优化属于他的神经网络
#         for net, opt, l_his in zip(nets, optimizers, losses_his):
#             output = net(b_x)              # get output for every net
#             loss = loss_func(output, b_y)  # compute loss for every net
#             opt.zero_grad()                # clear gradients for next train
#             loss.backward()                # backpropagation, compute gradients
#             opt.step()                     # apply gradients
#             l_his.append(loss.data.numpy())     # loss recoder

