import argparse



parser = argparse.ArgumentParser(description='Sequence Modeling - Copying Memory Task')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
#NLSTM gc = 0.25
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=8,
                    help='kernel size (default: 8)')
parser.add_argument('--iters', type=int, default=100,
                    help='number of iters per epoch (default: 100)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--blank_len', type=int, default=1000, metavar='N',
                    help='The size of the blank (i.e. T) (default: 1000)')
parser.add_argument('--seq_len', type=int, default=10,
                    help='initial history size (default: 10)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 50')
#NLSTM lr =1e-4 ok
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (deault: 5e-4)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 10)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
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
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from MCRM.copy_memory.utils import data_generator
from MCRM.copy_memory.model import *
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

batch_size = args.batch_size
seq_len = args.seq_len    # The size to memorize
epochs = args.epochs
iters = args.iters
T = args.blank_len
n_steps = T + (2 * seq_len)
n_classes = 10  # Digits 0 - 9
n_train = 10000
n_test = 1000

print(args)
print("Preparing data...")
train_x, train_y = data_generator(T, seq_len, n_train)
test_x, test_y = data_generator(T, seq_len, n_test)


channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout


if args.model == 'MCRM':
    model = MCRMModel(1, n_classes,intermidate=500, batch =args.batch_size)
elif args.model == 'NLSTM':
    model = NLSTM(1, n_classes,intermidate=448, batch =args.batch_size)
#['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']
elif args.model == 'LSTM':
    model = RNNModel('LSTM',1, n_classes,intermidate=900,batch =args.batch_size)
elif args.model == 'GRU':
    model = RNNModel('GRU',1, n_classes,intermidate=1050,batch =args.batch_size)
elif args.model == 'RNN_TANH':
    model = RNNModel('RNN_TANH',1, n_classes,intermidate=1800,batch =args.batch_size) 
elif args.model == 'RNN_RELU':
    model = RNNModel('RNN_RELU',1, n_classes,intermidate=1800,batch =args.batch_size)   
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
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

evaluation_time = 0

def evaluate():
    global evaluation_time
    global batch_size, seq_len, iters, epochs
    model.eval()
    total_loss = 0
    general_loss = 0 
    general_loss_count = 0
    start_time = time.time()
    correct = 0
    general_correct = 0 
    general_correct_counter = 0
    counter = 0
    change_batch_size(args.batch_size)
    start = time.time()
    for batch_idx, batch in enumerate(range(0, n_test, batch_size)):
        start_ind = batch
        end_ind = start_ind + batch_size
        #print(start_ind,end_ind)
        x = test_x[start_ind:end_ind]
        y = test_y[start_ind:end_ind]
        #print(len(x),len(test_x))
        if len(x) != args.batch_size:
            change_batch_size(len(x))
        #optimizer.zero_grad()
        out = model(x.unsqueeze(1).contiguous())
        loss = criterion(out.view(-1, n_classes), y.view(-1))
        pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        cn = pred.eq(y.data.view_as(pred)).cpu().sum()
        cnn =  out.view(-1, n_classes).size(0)
        correct += cn
        counter += cnn
        #if args.clip > 0:
        #   torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        #loss.backward()
        #optimizer.step()
        total_loss += loss.data[0]
        general_loss += loss.data[0]
        general_loss_count += 1
        general_correct += cn
        general_correct_counter += cnn
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('Test| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                ep, batch_idx, n_train // batch_size+1, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss, 100. * correct / counter))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0
        del x ,y,out,loss,pred,cn,cnn
    end = time.time()
    evaluation_time = (end - start)/len(test_x)    
    return general_loss/general_loss_count,100. * general_correct / general_correct_counter




def train(ep):
    global batch_size, seq_len, iters, epochs
    model.train()
    total_loss = 0
    general_loss = 0 
    general_loss_count = 0
    start_time = time.time()
    correct = 0
    general_correct = 0 
    general_correct_counter = 0
    counter = 0
    change_batch_size(args.batch_size)
    for batch_idx, batch in enumerate(range(0, n_train, batch_size)):
        start_ind = batch
        end_ind = start_ind + batch_size

        x = train_x[start_ind:end_ind]
        y = train_y[start_ind:end_ind]
        if len(x) != args.batch_size:
            change_batch_size(len(x))
        optimizer.zero_grad()
        out = model(x.unsqueeze(1).contiguous())
        loss = criterion(out.view(-1, n_classes), y.view(-1))
        pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        cn = pred.eq(y.data.view_as(pred)).cpu().sum()
        cnn =  out.view(-1, n_classes).size(0)
        correct += cn
        counter += cnn
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
        general_loss += loss.data[0]
        general_loss_count += 1
        general_correct += cn
        general_correct_counter += cnn
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('Train| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                ep, batch_idx, n_train // batch_size+1, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss, 100. * correct / counter))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0
        del x ,y,out,loss,pred,cn,cnn
            
    return general_loss/general_loss_count,100. * general_correct / general_correct_counter


for ep in range(1, epochs + 1):
    tloss,tacc = train(ep)
    vloss,vacc = evaluate()
    errorslog = {'training_loss':tloss,'training_acc':tacc,
              'validation_loss':vloss,'validation_acc':vacc}
    print(errorslog)
    logger.log(errorslog)
paradic = {'parameters':_params,'evaluation_time':evaluation_time}
logger.log(paradic)    
logger.finish_log()