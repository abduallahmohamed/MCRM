import numpy as np 
import pickle
import math 

class LightDeepLearningLogger:
    
    def __init__(self,exp_name,path):
        
        self.exp_name = exp_name
        self.path = path
        self.data_log = []
        
    def log(self,data):
        self.data_log.append(data)
        
    def finish_log(self):
        output = open(self.path + self.exp_name +'.pkl', 'wb')
        pickle.dump(self.data_log, output)

        output.close()

#https://stackoverflow.com/questions/23787296/how-do-i-format-a-number-in-python-with-k-for-thousand-etc-for-arbitrary-preffix
def num_fmt(num):
    i_offset = 15 # change this if you extend the symbols!!!
    prec = 3
    fmt = '.{p}g'.format(p=prec)
    symbols = ['Y', 'T', 'G', 'M', 'k', '', 'm', 'u', 'n']

    e = math.log10(abs(num))
    if e >= i_offset + 3:
        return '{:{fmt}}'.format(num, fmt=fmt)
    for i, sym in enumerate(symbols):
        e_thresh = i_offset - 3 * i
        if e >= e_thresh:
            return '{:{fmt}}{sym}'.format(num/10.**e_thresh, fmt=fmt, sym=sym)
    return '{:{fmt}}'.format(num, fmt=fmt)
    
def count_parameters(model):
    s = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_fmt(s)