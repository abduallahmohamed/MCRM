import argparse


parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
#NLSTM clip = 0.1
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=200,
                    help='sequence length (default: 200)')
#NLSTM lr = 0.01
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate (default: 0.001)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--model', default='MCRM',
                    help='MCRM,NLSTM,LSTM,GRU,RNN_TANH,RNN_RELU')
parser.add_argument('--gpu', default='0',
                    help='Which GPU to use')
parser.add_argument('--v', default='0',
                    help='version')  
parser.add_argument('--lgpath', default='./Addinglog/',
                    help='log path')   
args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import numpy as np  
import torch
import torch.optim as optim
import torch.nn.functional as F
from MCRM.adding_problem.utils import data_generator
from MCRM.adding_problem.model import NLSTM , MCRMModel , RNNModel
from MCRM.general_utilities import LightDeepLearningLogger , count_parameters
import time


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


input_channels = 2
n_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs

print(args)
print("Producing data...")
X_train, Y_train = data_generator(50000, seq_length)
X_test, Y_test = data_generator(1000, seq_length)


# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout

if args.model == 'MCRM':
    model = MCRMModel(input_channels, n_classes,intermidate=300, batch =args.batch_size)
#['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']
elif args.model == 'NLSTM':
    model = NLSTM(input_channels, n_classes,intermidate=77, batch =args.batch_size)
elif args.model == 'LSTM':
    model = RNNModel('LSTM',input_channels, n_classes,intermidate=153,batch =args.batch_size)
elif args.model == 'GRU':
    model = RNNModel('GRU',input_channels, n_classes,intermidate=177,batch =args.batch_size)
elif args.model == 'RNN_TANH':
    model = RNNModel('RNN_TANH',input_channels, n_classes,intermidate=308,batch =args.batch_size) 
elif args.model == 'RNN_RELU':
    model = RNNModel('RNN_RELU',input_channels, n_classes,intermidate=308,batch =args.batch_size)   
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
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    complete_loss = 0 
    complete_loss_div = 0
    change_batch_size(args.batch_size)

    for i in range(0, X_train.size()[0], batch_size):
        if i + batch_size > X_train.size()[0]:
            x, y = X_train[i:], Y_train[i:]
            change_batch_size(len(x))
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        #print("Input vector",x.size())
        #print("Target Vector",y.size())
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1


        cur_loss = loss.data[0]
        complete_loss += cur_loss
        complete_loss_div += 1
        processed = min(i+batch_size, X_train.size()[0])
        if batch_idx % 20 == 0:
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, cur_loss))
            
    return complete_loss/complete_loss_div

evaluation_time = 0
def evaluate():
    global evaluation_time
    model.eval()
    #print(X_test.size())
    change_batch_size(len(X_test))

    start = time.time()
    output = model(X_test)
    end = time.time()
    evaluation_time = (end - start)/len(X_test)
    test_loss = F.mse_loss(output, Y_test)
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.data[0]))
    return test_loss.data[0]


for ep in range(1, epochs+1):

    trainingloss = train(ep)
    tloss = evaluate()
    errors = {'training_loss':trainingloss,'testing_loss':tloss}
    logger.log(errors)

paradic = {'parameters':_params,'evaluation_time':evaluation_time}
logger.log(paradic)    
logger.finish_log()




