# Runs in Ubuntu 18.04 / Python / Pytorch / CUDA
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import argparse
import os 
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Process some parameters')
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--data_dir',type=str,default='./dataset')
parser.add_argument('--num_workers',type=int,default=2)
parser.add_argument('--num_epoches',type=int,default=40)
parser.add_argument('--lr',type=float,default=1e-2)
parser.add_argument('--save_dir',type=str,default='./save_dir')
parser.add_argument('--model',type=str,default='mnasnet0_5',help="Network name")
parser.add_argument('--pretrain',type=bool,default=True,help="The model is set unpretrained as default.")
parser.add_argument('--num_classes',type=int,default=10,help=".")
parser.add_argument('--use_cuda',type=bool,default=True,help='Whether to use cuda')
parser.add_argument('--evaluate',type=bool,default=False,help='Whether to use cuda')

args = parser.parse_args()

data_dir = args.data_dir
batch_size = args.batch_size
learning_rate = args.lr
num_workers = args.num_workers
save_dir = args.save_dir+'_'+args.model
num_classes = args.num_classes
num_epoches = args.num_epoches

print("Run config:\ndata_dir:{}\nbatch_size:{}\nlr:{}\n num_workers:{}\nsave_dir:{}\n num_classes:{}\n".format(args.data_dir,args.batch_size,args.lr,args.num_workers,save_dir,args.num_classes))
print("Model:{}\nPretrain:{}\n".format(args.model,args.pretrain))

# todo: data pre-process: format to png / rename in order 

def save_checkpoint(state, epoch, root):
    filename = 'checkpoint_%03d.pth.tar' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

# def adjust_learning_rate(optimizer, epoch, args, method='cosine'):
#     if method == 'cosine':
#         T_total = float(args.epochs)
#         T_cur = float(epoch)
#         lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
#     elif method == 'multistep':
#         lr = args.lr
#         for epoch_step in args.lr_steps:
#             if epoch >= epoch_step:
#                 lr = lr * 0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     str_lr = '%.6f' % (lr)
#     return str_lr



# class CrossEntropyLabelSmooth(nn.Module):
#     """
#         label smooth
#     """
#     def __init__(self, num_classes, epsilon):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.logsoftmax = nn.LogSoftmax(dim=1)

#     def forward(self, inputs, targets):
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = (-targets * log_probs).mean(0).sum()
#         return loss



class Averagvalue(object):
    """Computes and stores the average and current value"""


    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.list.append(val)

train_loss = Averagvalue()
train_acc = Averagvalue()
test_loss = Averagvalue()
test_acc = Averagvalue()

if os.path.exists(save_dir) is not True:
    os.mkdir(save_dir)

# Dataset / dataloader 
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
                transforms.Scale(288),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean = mean, std = std),
                ])

train_data = datasets.ImageFolder(data_dir,transform=transform)

# Hold-out method to divide 
print("train: ",len(train_data))

train_data, val_data = torch.utils.data.random_split(train_data,[int(len(train_data)*0.8),len(train_data)-int(len(train_data)*0.8)])

print("train:{} \t val:{} \n".format(len(train_data),len(val_data)))

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)

if (args.model=='resnet18'): # doing well
    model = models.resnet18(pretrained=args.pretrain)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)
# elif (args.model=='alexnet'):
#     model = models.alexnet(pretrained=args.pretrain)
# elif (args.model=='vgg16'):
#     model = models.vgg16(pretrained=args.pretrain)
# elif (args.model=='densenet161'):
#     model = models.densenet161(pretrained=args.pretrain)
# elif (args.model=='inception_v3'):
#     model = models.inception_v3(pretrained=args.pretrain)
# elif (args.model=='googlenet'):
#     model = models.googlenet(pretrained=args.pretrain)
# elif (args.model=='resnext50_32x4d'):
#     model = models.resnext50_32x4d(pretrained=args.pretrain)
# elif (args.model=='wide_resnet50_2'):
#     model = models.wide_resnet50_2(pretrained=args.pretrain)
elif (args.model=='mobilenet_v2'): # not doing well 
    model = models.mobilenet_v2(pretrained=args.pretrain)
    model.classifier[1].out_features = num_classes
elif (args.model=='squeezenet1_0'): # doing well
    model = models.squeezenet1_0(pretrained=args.pretrain)
    model.classifier[1] = nn.Conv2d(512,num_classes,kernel_size=(1,1),stride=(1,1))
    model.num_classes=num_classes  
elif (args.model=='mnasnet0_5'): # not doing well
    model = models.mnasnet0_5(pretrained=args.pretrain)
    model.classifier[1].out_features = num_classes
# elif (args.model=='shufflenet_v2_x0_5'):
#     model = models.shufflenet_v2_x0_5(pretrained=args.pretrain)

else:
    print("Wrong model name!\n")
    exit()

# print(model.fc)


if args.use_cuda:
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    print('\ncuda is used, with %d gpu devices \n' % torch.cuda.device_count())
else:
    print('\n cuda is not used, the running might be slow\n')

use_gpu = torch.cuda.is_available()
loss_fn = nn.CrossEntropyLoss()
# loss_fn_smooth = CrossEntropyLabelSmooth(args.num_classes, 0.1)
# loss_fn_smooth = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)

optimizer = optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(num_epoches):
    print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)      # .format为输出格式，formet括号里的即为左边花括号的输出
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader):
        img, label = data
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img) # Variable()?
        label = Variable(label)
        # 向前传播
        out = model(img)
        loss = loss_fn(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)     # 预测最大值所在的位置标签
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_data)), running_acc / (len(train_data))))
    train_acc.update(running_acc / (len(train_data)))
    train_loss.update(running_loss / (len(train_data)))  

    model.eval()    # 模型评估
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for data in val_loader:      # 测试模型 why  no enumerate here
            img, label = data
            if use_gpu:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = loss_fn(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        val_data)), eval_acc / (len(val_data))))
    test_acc.update(eval_acc / (len(val_data)))
    test_loss.update(eval_loss / (len(val_data))) 
    # print()
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
         }, epoch, save_dir)


print("\ntrain_acc:",train_acc.list)
print("\ntrain_loss:",train_loss.list)
print("\ntest_acc:",test_acc.list)
print("\ntest_loss:",test_loss.list)

# todo: input a new pic to classify / evaluate 

