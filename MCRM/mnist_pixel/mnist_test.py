
import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 60)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
parser.add_argument('--model', default='MCRM',
                    help='MCRM,NLSTM,LSTM,GRU,RNN_TANH,RNN_RELU')
parser.add_argument('--gpu', default='0',
                    help='Which GPU to use')
parser.add_argument('--v', default='0',
                    help='version')  
parser.add_argument('--lgpath', default='./Playground/',
                    help='log path')   
                    
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from MCRM.mnist_pixel.utils import data_generator
from MCRM.mnist_pixel.model import *
from MCRM.general_utilities import LightDeepLearningLogger , count_parameters
import time
import numpy as np

#Random seed settings 
torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
log_path = args.lgpath
if not os.path.exists(log_path):
    os.makedirs(log_path)

EXP_NAME = args.model+'_'+args.v
logger = LightDeepLearningLogger(EXP_NAME,log_path)
print('*'*30)
print(EXP_NAME)
print('*'*30)

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
if args.model == 'MCRM':
    model = MCRMModel(input_channels, n_classes,intermidate=108, batch =args.batch_size)
elif args.model == 'NLSTM':
    model = NLSTM(input_channels, n_classes,intermidate=97, batch =args.batch_size)

#['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']
elif args.model == 'LSTM':
    model = RNNModel('LSTM',input_channels, n_classes,intermidate=192,batch =args.batch_size)
elif args.model == 'GRU':
    model = RNNModel('GRU',input_channels, n_classes,intermidate=222,batch =args.batch_size)
elif args.model == 'RNN_TANH':
    model = RNNModel('RNN_TANH',input_channels, n_classes,intermidate=384,batch =args.batch_size) 
elif args.model == 'RNN_RELU':
    model = RNNModel('RNN_RELU',input_channels, n_classes,intermidate=384,batch =args.batch_size)   
else:
    raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['MCRM,'NLSTM,'TCN','LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")    
_params =    count_parameters(model)                              
print("Total parameters of ",args.model,_params)

def change_batch_size(new_batch):
    if args.model == 'TCN':
        pass
    else:
        model.init_state(new_batch)
        
if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(ep):
    global steps
    train_loss = 0
    correct = 0
    correct_count = len(train_loader.dataset)
    general_loss = 0 
    general_loss_count = 0
    general_correct = 0 
    general_correct_counter = 0
    change_batch_size(args.batch_size)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            #print('Permute')
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        if len(data) != args.batch_size:
            print('Changing bsize', len(data))
            change_batch_size(len(data))
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.data[0]
        general_loss += loss.data[0]
        general_loss_count += 1
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss/args.log_interval, steps))
            train_loss = 0
        del data,target,output,pred,loss
    return  general_loss/general_loss_count,100. * correct/correct_count

evaluation_time = 0

def test():
    global evaluation_time
    model.eval()
    test_loss = 0
    correct = 0
    start = time.time()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data, volatile=True), Variable(target)
        if len(data) != args.batch_size:
            change_batch_size(len(data))
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        del data,target,output,pred
    end = time.time()
    evaluation_time = (end - start)/len(test_loader.dataset)    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss,100. * correct / len(test_loader.dataset)


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        tloss,tacc = train(epoch)
        vloss,vacc = test()
        errorslog = {'training_loss':tloss,'training_acc':tacc,
              'validation_loss':vloss,'validation_acc':vacc}
        print(errorslog)
        logger.log(errorslog)
        if epoch % 30 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
paradic = {'parameters':_params,'evaluation_time':evaluation_time}
logger.log(paradic)    
logger.finish_log()