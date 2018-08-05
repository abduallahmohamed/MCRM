import sys
sys.path.append("../../")
from torch import nn
from MCRM.mcrm import MCRM , GRUCell
from MCRM.nlstm import NestedLSTM
from torch.autograd import Variable

###################################################################
################################MCRM ############################
###################################################################        
class MCRMModel(nn.Module):
    def __init__(self, input_size, output_size,intermidate=89,batch=32, dropout=0):
        super(MCRMModel, self).__init__()
        self.batch = batch
        self.intermidate= intermidate
        

        self.mglstm = MCRM(input_size,self.intermidate)
        
        self.linear = nn.Linear(self.intermidate, output_size)

        
        self.init_state(self.batch)
        self.init_weights()


    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def init_state(self,batch):
        self.state = self.mglstm.init_hidden(batch)
    def repackage_hidden(self):
        """Wraps hidden states in new Variables, to detach them from their history."""
        h_new = []

        for i in range(len(self.state)):
            main,nested = self.state[i]
            m_h,m_o = main
            n_h = nested
            #(cell_state,self.cell_core.init_hidden(batch_size))
            n_main = (Variable(m_h.data),Variable(m_o.data))
            n_nested = (Variable(n_h.data))

            h_new.append((n_main,n_nested))
            
        return h_new
    def forward(self, x):
        self.state = self.repackage_hidden()
        x = x.permute(0,2,1) # B, F,S --> B,S,F
        layer_output_list, last_state_list, self.state = self.mglstm(x,self.state)
        #print("lSTM output" ,layer_output_list[0].size())
        x = layer_output_list[0][:,:,:]
        #x=x.transpose(1, 2)
        #print("OP", x.size())
        x = self.linear(x)
        #print("Output size)",x.size())
        return x
###################################################################
################################NLSTM #############################
###################################################################        
class NLSTM(nn.Module):
    def __init__(self, input_size, output_size,intermidate=89,batch=32, dropout=0):
        super(NLSTM, self).__init__()
        self.batch = batch
        self.intermidate= intermidate


        self.nlstm = NestedLSTM(input_size,self.intermidate)
        
        self.linear = nn.Linear(self.intermidate, output_size)

        
        self.init_state(self.batch)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        
    def init_state(self,batch):
        self.state = self.nlstm.init_hidden(batch)
    def repackage_hidden(self):
        """Wraps hidden states in new Variables, to detach them from their history."""
        h_new = []

        for i in range(len(self.state)):
            main,nested = self.state[i]
            m_h,m_o = main
            n_h,n_o = nested
            #(cell_state,self.cell_core.init_hidden(batch_size))
            n_main = (Variable(m_h.data),Variable(m_o.data))
            n_nested = (Variable(n_h.data),Variable(n_o.data))

            h_new.append((n_main,n_nested))
            
        return h_new
    def forward(self, x):
        self.state = self.repackage_hidden()
        x = x.permute(0,2,1) # B, F,S --> B,S,F
        layer_output_list, last_state_list, self.state = self.nlstm(x,self.state)
        #print("lSTM output" ,layer_output_list[0].size())
        x = layer_output_list[0][:,:,:]
        #x=x.transpose(1, 2)
        #print("OP", x.size())
        x = self.linear(x)
        #print("Output size)",x.size())
        return x
        

###################################################################
################################LSTM,RNN, GRU #####################
###################################################################
class RNNModel(nn.Module):
    """Container module with a recurrent module, and a decoder."""

    def __init__(self, rnn_type, input_size,output_size,intermidate=89, batch=32,dropout=0):
        super(RNNModel, self).__init__()
        self.batch = batch
        self.intermidate = intermidate #This variable is tuned to make sure that we have the same number of parameters
        self.rnn_type = rnn_type


        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, self.intermidate, 1,batch_first= True) #1 = Number of layers 
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, self.intermidate, 1, nonlinearity=nonlinearity,batch_first = True)
            
        self.linear = nn.Linear(self.intermidate, output_size)


        self.init_weights()
        self.init_state(self.batch)
    def repackage_hidden(self,v):
    
        ###Wraps hidden states in new Variables, to detach them from their history###
        if type(v) == Variable:
            return Variable(v.data)
        else:
            return tuple(self.repackage_hidden(v) for v in self.state)
            

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        self.state = self.repackage_hidden(self.state)
        x = x.permute(0,2,1) # B, F,S --> B,S,F
        x, self.state = self.rnn(x, self.state)
        #print("RNN output" ,x.size())
        x = x[:,:,:]
        #print("Linear input", x.size())
        x = self.linear(x)
        #print("Output size)",x.size())
        return x
        #output, hidden = self.rnn(emb, hidden)
        #output = self.drop(output)
        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
    def init_state(self,batch):
        self.state = self.init_hidden(batch)
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(1, bsz, self.intermidate).zero_()).cuda(),
                    Variable(weight.new(1, bsz, self.intermidate).zero_()).cuda())
        else:
            return Variable(weight.new(1, bsz, self.intermidate).zero_().cuda())
  