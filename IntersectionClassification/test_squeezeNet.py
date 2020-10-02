# Runs in windows 10 / no gpu or cuda needed 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import argparse
import os 
from torch.autograd import Variable
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Process some parameters')
parser.add_argument('--batch_size',type=int,default=12)
parser.add_argument('--data_dir',type=str,default='./test')
parser.add_argument('--num_workers',type=int,default=2)
# parser.add_argument('--num_epoches',type=int,default=40)
# parser.add_argument('--lr',type=float,default=1e-2)
parser.add_argument('--save_dir',type=str,default='./save_dir',help="the place to save checkpoint")
parser.add_argument('--model',type=str,default='squeezenet1_0',help="Network name")
# parser.add_argument('--pretrain',type=bool,default=True,help="The model is set unpretrained as default.")
parser.add_argument('--num_classes',type=int,default=10,help=".")
parser.add_argument('--use_cuda',type=bool,default=True,help='Whether to use cuda')
parser.add_argument('--evaluate',type=bool,default=None,help='Whether to use cuda')

args = parser.parse_args()

data_dir = args.data_dir
batch_size = args.batch_size
# learning_rate = args.lr
num_workers = args.num_workers
save_dir = args.save_dir+'_'+args.model
num_classes = args.num_classes
# num_epoches = args.num_epoches


print("Test run config:\ndata_dir:{}\nbatch_size:{}\n num_workers:{}\nsave_dir:{}\n num_classes:{}\n".format(args.data_dir,args.batch_size,args.num_workers,args.save_dir,args.num_classes))
print("Model:{}\n".format(args.model))

def load_checkpoint(args):
# def load_checkpoint(args, running_file):
    model_dir = os.path.join(args.save_dir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''
    if args.evaluate is not None:
        model_filename = os.path.join(model_dir, args.evaluate)
    elif os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()

    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        if not args.use_cuda:
            print("Not using cuda, may cause problem for comment.")
            # state['state_dict'] = ajust_state(state['state_dict'])
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    # running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
    # running_file.flush()

    return state


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

test_loss = Averagvalue()
test_acc = Averagvalue()
print(save_dir)
if os.path.exists(save_dir) is not True:
    print("Error! Please check your path of checkpoint.")
    exit()
    # break

# Dataset / dataloader 
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
                transforms.Scale(288),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean = mean, std = std),
                ])

test_data = datasets.ImageFolder(data_dir,transform=transform)

# # ResNet18
# model = models.resnet18()
# num_ftrs = model.fc.in_features 
# model.fc = nn.Linear(num_ftrs,num_classes) 

# SqueezeNet
model = models.squeezenet1_0()
model.classifier[1] = nn.Conv2d(512,num_classes,kernel_size=(1,1),stride=(1,1))
model.num_classes=num_classes  

# # MnasNet 
# model = models.mnasnet0_5()
# model.classifier[1].out_features = num_classes  

# todo 
if args.use_cuda:
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    print('\ncuda is used, with %d gpu devices \n' % torch.cuda.device_count())
else:
    print('\n cuda is not used, the running might be slow\n')


# checkpoint = torch.load("./save_dir_resnet18/save_models/checkpoint_005.pth.tar")
checkpoint = torch.load("./save_dir_squeezenet1_0/save_models/checkpoint_007.pth.tar")
# for trail 4 
# 007 20/20 
# 009 19/20 
# 013 2/20 
# 034 20/20 

# for trail 3
# 006 9/20 
# 004 8/20
# 009 10/20
# 011 8/20

# for trail 2 
# ./save_dir_squeezenet1_0/save_models/checkpoint_010.pth.tar 8/20 
# 009 9/20
# 008 8/20
# 020 6/20
# 015 7/20
# 012 7/20
# 013 7/20
 # NO MNAS NET
# checkpoint = torch.load("./save_dir_mnasnet0_5/save_models/checkpoint_025.pth.tar")
# checkpoint = torch.load("./checkpoint.pth.tar",map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['state_dict'])

# the sequence is not natural number, it's like 1,10,11,2,20,3,4,5,...
print(test_data.__len__())
print(test_data.imgs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f = open('./test.dat',"wb")


for img,label in test_data: # default it should be in order
    # out = model(img)
    img = img.to(device)
    out = model(img[None, ...])
    _, pred = torch.max(out, 1) 
    # line = str(int(label)+1)+" "+str(int(pred))+'\n'
    line = str(int(pred))+'\n'

    # print(type(line))
    # print(line)
    f.write(str.encode(line))
    # f.write(str(pred.int())+'\n')
    # print(out)

    # print(type(pred))
    # print(int(pred))

    # print(int(label)+1,int(pred))
    
    print(label,pred)

    # Checked, in the sequence as shown above. 
    # plt.imshow(img[0])
    # plt.show()

    # print(img)
f.close()

# for i in range(test_data.__len__()):
#     result = model(test_data[i][0])


# print(test_data)

# # todo: Sequence / file name 
# test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)

# if (args.model=='resnet18'): # doing well
#     model = models.resnet18()
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs,num_classes)
#     # model.load_state_dict(torch.load()) 

# elif (args.model=='mobilenet_v2'): # not doing well 
#     model = models.mobilenet_v2()
#     model.classifier[1].out_features = num_classes
# elif (args.model=='squeezenet1_0'): # doing well
#     model = models.squeezenet1_0()
#     model.classifier[1] = nn.Conv2d(512,num_classes,kernel_size=(1,1),stride=(1,1))
#     model.num_classes=num_classes  
# elif (args.model=='mnasnet0_5'): # not doing well
#     model = models.mnasnet0_5()
#     model.classifier[1].out_features = num_classes

# else:
#     print("Wrong model name!\n")
#     exit()

# if args.use_cuda:
#     model = torch.nn.DataParallel(model).cuda()
#     print('\ncuda is used, with %d gpu devices \n' % torch.cuda.device_count())
# else:
#     print('\n cuda is not used, the running might be slow\n')

# checkpoint = load_checkpoint(args)
# # checkpoint = load_checkpoint(args, running_file)
# if checkpoint is not None:
#     # args.start_epoch = checkpoint['epoch'] + 1
#     model.load_state_dict(checkpoint['state_dict'])
#     # optimizer.load_state_dict(checkpoint['optimizer'])
# else:
#     print('Error! Wrong checkpoint file.')
#     exit()
#     # break


# use_gpu = torch.cuda.is_available()
# loss_fn = nn.CrossEntropyLoss()
# # optimizer = optim.SGD(model.parameters(),lr=learning_rate)

# model.eval()    # 模型评估
# eval_loss = 0
# eval_acc = 0
# # todo: no label actually
# with torch.no_grad():
#     for data in test_loader:      
#         img, label = data
#         if use_gpu:
#             img = Variable(img, volatile=True).cuda()
#             label = Variable(label, volatile=True).cuda()
#         else:
#             img = Variable(img, volatile=True)
#             label = Variable(label, volatile=True)
#         out = model(img)
#         print("out: ",out)
#         loss = loss_fn(out, label)
#         eval_loss += loss.item() * label.size(0)
#         _, pred = torch.max(out, 1)
#         num_correct = (pred == label).sum()
#         eval_acc += num_correct.item()
# # todo: delete later         
# print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#     test_data)), eval_acc / (len(test_data))))
# test_acc.update(eval_acc / (len(test_data)))
# test_loss.update(eval_loss / (len(test_data)))
# # f.close
# # test: with 8 classes 


