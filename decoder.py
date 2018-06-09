import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import h5py
import torch.nn.functional as F

sentRNN_lstm_dim = 512 # the sentence LSTM hidden units
sentRNN_FC_dim = 1024 # the fully connected units
wordRNN_lstm_dim = 512 # the word LSTM hidden units
word_embed_dim = 1024 # the learned embedding vectors for the words

class HierRNN(nn.Module):
    def __init__(self, n_words,
                       batch_size,
                       num_boxes,
                       feats_dim,
                       project_dim,
                       sentRNN_lstm_dim,
                       sentRNN_FC_dim,
                       wordRNN_lstm_dim,
                       S_max,
                       N_max,
                       word_embed_dim,bias_init_vector=None):
        """Set the hyper-parameters and build the layers."""
        super(HierRNN, self).__init__()
        self.regionPooling_W = (-0.1 - 0.1) * torch.rand(feats_dim, project_dim) + 0.1
        self.regionPooling_b = torch.zeros(project_dim)
        self.logistic_Theta_W = torch.rand(sentRNN_lstm_dim,2)*0.2-0.1
        self.logistic_Theta_b = torch.zeros(2)
        
        self.fcregionPool = nn.Linear(feats_dim,project_dim)
        self.maxPool50 = nn.MaxPool1d(50)
        self.batch_size = batch_size
        #embedding
        self.S_max = S_max
        self.N_max = N_max
        self.embed = nn.Embedding(n_words, word_embed_dim)
        self.sentLSTM = nn.LSTM(project_dim, sentRNN_lstm_dim,batch_first=True)
        
        self.word_LSTM = nn.LSTM(word_embed_dim,wordRNN_lstm_dim,2,batch_first=True)
        
        self.sent_FC = nn.Linear(sentRNN_lstm_dim,sentRNN_FC_dim)
        self.sent_FC2  = nn.Linear(sentRNN_FC_dim,1024)
        self.word_FC = nn.Linear(wordRNN_lstm_dim,n_words)
        self.binarize = nn.Linear(sentRNN_lstm_dim,2)
        self.m = nn.Softmax(dim=2)
#         self.sentLstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         self.init_weights()


    def init_hidden(self, batch_size):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, sentRNN_lstm_dim).zero_())
        
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        #feats  = 
        #print captions[1,:,:]
        sent_state = self.init_hidden(self.batch_size)
        
#         features = self.fcregionPool(features)
#         features.transpose_(1,2)
        #print features
        features = features.view(self.batch_size ,1,-1)
        temp_distribution= Variable(torch.Tensor()).cuda()
        final_output = Variable(torch.Tensor()).cuda()
        for i in range(0,self.S_max):
            sent_output,sent_state = self.sentLSTM(features)
            hidden1 = F.relu(self.sent_FC(sent_output))
            sent_topic_vec = F.relu(self.sent_FC2(hidden1))
            
            sentRNN_binary = self.binarize(sent_output)
            sentRNN_binary = sentRNN_binary
            h1 = sent_topic_vec[:,:,0:512].transpose_(0,1)
            c1 = sent_topic_vec[:,:,512:].transpose_(0,1)
            h1= torch.cat((h1,h1),0)
            c1= torch.cat((c1,c1),0)
            #[000001111]
            temp_distribution = torch.cat((temp_distribution,sentRNN_binary),1)
            
            ############not sure 
            temp_output=Variable(torch.Tensor()).cuda()
#             for j in range(0, self.N_max):
                
#                 current_embed = self.embed(captions[:,i,j])
#                 print current_embed.size()
                
#                 current_embed=current_embed.unsqueeze(1)
#                 word_output,word_state = self.word_LSTM(current_embed,(h1,c1))
#                 word_output = self.m(self.word_FC(word_output))
#                 temp_output = torch.cat((temp_output,word_output),1)
            current_embed = self.embed(captions[:,i,:])
            #print(current_embed.size())
        
            #need to do pack padded sequence. 
            
            word_output,word_state = self.word_LSTM(current_embed,(h1,c1))
            #convert it back
            
            word_output = self.word_FC(word_output)
            word_output= word_output.unsqueeze(1)
            final_output = torch.cat((final_output,word_output),1)
        #print final_output.size()
        return temp_distribution,final_output
    #input dimemtion [batch*1024]
    def sample(self, features,captionMask=None,states=None):
        """Decode image feature vectors and generates captions."""
        #feats  = 
       
        #sent_state = self.init_hidden(self.batch_size)
        
#         features = self.fcregionPool(features)
#         features.transpose_(1,2)
        #print features
    
    
        features = features.view(self.batch_size ,1,-1)
        temp_distribution= Variable(torch.Tensor()).cuda()
        final_output = Variable(torch.Tensor()).cuda()
        for i in range(0,self.S_max):
            sent_output,sent_state = self.sentLSTM(features)
            hidden1 = F.relu(self.sent_FC(sent_output))
            sent_topic_vec = F.relu(hidden1)
            
#             sentRNN_binary = self.binarize(sent_output)
#             sentRNN_binary = self.m(sentRNN_binary)
            h1 = sent_topic_vec[:,:,0:512].transpose_(0,1)
            c1 = sent_topic_vec[:,:,512:].transpose_(0,1)
            h1= torch.cat((h1,h1),0)
            c1= torch.cat((c1,c1),0)
            #[000001111]
#             temp_distribution = torch.cat((temp_distribution,sentRNN_binary),1)
            
            temp_output=[]
            c_embed = Variable(torch.Tensor()).cuda()
            for j in range(0, self.N_max):
                #current_embed = self.embed(captions[:,i,j])
                #print current_embed.size()
                if j==0:
                    inputs = Variable(torch.zeros([self.batch_size,1,1]))
                    current_embed = self.embed(inputs)
                else:
                    current_embed = c_embed
                #current_embed=current_embed.unsqueeze(1)
                word_output,word_state = self.word_LSTM(current_embed,(h1,c1))
                word_output = self.m(self.word_FC(word_output.squeeze(1)))
                _, predicted = outputs.max(1) 
                
                temp_output.append(predicted)
                c_embed = self.embed(predicted)
                #inputs = inputs.unsqueeze(1) 
            
            temp_output= temp_output.unsqueeze(1)
            final_output = torch.cat((final_output,temp_output),1)
        #print final_output.size()
        return temp_distribution,final_output