import argparse



parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--emsize', type=int, default=600,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=40,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
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
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
from MCRM.word_ptb.utils import *
from MCRM.word_ptb.model import *
from MCRM.general_utilities import LightDeepLearningLogger , count_parameters
import pickle
from random import randint
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

print(args)
corpus = data_generator(args)
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, eval_batch_size, args)


n_words = len(corpus.dictionary)

num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
tied = args.tied


if args.model == 'MCRM':
    model = MCRMModel(args.emsize, n_words,intermidate=500, batch =args.batch_size, dropout=dropout, emb_dropout=emb_dropout, tied_weights=tied)
elif args.model == 'NLSTM':
    model = NLSTM(args.emsize, n_words,intermidate=475, batch =args.batch_size, dropout=dropout, emb_dropout=emb_dropout, tied_weights=tied)
#['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']
elif args.model == 'LSTM':
    model = RNNModel('LSTM',args.emsize, n_words,intermidate=620,batch =args.batch_size, dropout=dropout, emb_dropout=emb_dropout, tied_weights=tied)
elif args.model == 'GRU':
    model = RNNModel('GRU',args.emsize, n_words,intermidate=662,batch =args.batch_size, dropout=dropout, emb_dropout=emb_dropout, tied_weights=tied)
elif args.model == 'RNN_TANH':
    model = RNNModel('RNN_TANH',args.emsize, n_words,intermidate=800,batch =args.batch_size, dropout=dropout, emb_dropout=emb_dropout, tied_weights=tied) 
elif args.model == 'RNN_RELU':
    model = RNNModel('RNN_RELU',args.emsize, n_words,intermidate=800,batch =args.batch_size, dropout=dropout, emb_dropout=emb_dropout, tied_weights=tied)   
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

# May use adaptive softmax to speed up training
criterion = nn.CrossEntropyLoss()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

evaluation_time = 0

def evaluate(data_source):
    global evaluation_time

    model.eval()
    total_loss = 0
    processed_data_size = 0
    start = time.time()

    for i in range(0, data_source.size(1) - 1, args.validseqlen):
        if i + args.seq_len - args.validseqlen >= data_source.size(1) - 1:
            continue
        data, targets = get_batch(data_source, i, args, evaluation=True)
        if len(data) != args.batch_size:
            change_batch_size(len(data))
        output = model(data)

        # Discard the effective history, just like in training
        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        final_target = targets[:, eff_history:].contiguous().view(-1)

        loss = criterion(final_output, final_target)

        # Note that we don't add TAR loss here
        total_loss += (data.size(1) - eff_history) * loss.data
        processed_data_size += data.size(1) - eff_history
    end = time.time()
    evaluation_time = (end - start)/len(data_source)
    return total_loss[0] / processed_data_size


def train():
    # Turn on training mode which enables dropout.
    global train_data
    model.train()
    total_loss = 0
    losses = []
    start_time = time.time()
    change_batch_size(args.batch_size)

    for batch_idx, i in enumerate(range(0, train_data.size(1) - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, args)
        if len(data) != args.batch_size:
            change_batch_size(len(data))
        optimizer.zero_grad()
        output = model(data)

        # Discard the effective history part
        eff_history = args.seq_len - args.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(final_output, final_target)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data[0]

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss/ args.log_interval
            losses.append(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, train_data.size(1) // args.validseqlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return sum(losses) * 1.0 / len(losses)

if __name__ == "__main__":
    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = train()
            val_loss = evaluate(val_data)
            test_loss = evaluate(test_data)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 5 and val_loss >= max(all_vloss[-10:]): #It was 5 orignially
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            errorslog = {'training_loss':train_loss,'training_ppl':math.exp(train_loss),
                      'validation_loss':val_loss,'validation_ppl':math.exp(val_loss),
                      'testing_loss':test_loss,'testing_ppl':math.exp(test_loss)}
            logger.log(errorslog)
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    paradic = {'parameters':_params,'evaluation_time':evaluation_time}
    logger.log(paradic)    
    logger.finish_log()
