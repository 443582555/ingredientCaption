
# coding: utf-8

# In[1]:


#Python Packages.
import os
import time
import random
import numpy as np

# Torch Packages
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn

# Import the Model.
from model import image_ingredient
from decoder import HierRNN
from data_loader import ImageLoader
import pdb
# Packages which will help Loading the data.
import pickle
import lmdb
from PIL import Image


# In[2]:


IMG_PATH = '/home/yifu/Documents/Mycode/python/hierarchicalRNN/jasha/'
# GIve the path for the LMDB files that were created.
DATA_PATH = '/home/yifu/Documents/Mycode/python/hierarchicalRNN/jasha/lmdb/'
WORKERS = 0
BATCH_SIZE = 40
VAL_FREQ = 1
START_EPOCH = 0
TOTAL_EPOCH = 500


#batch_size = 50 # Being support batch_size
num_boxes = 50 # number of Detected regions in each image
feats_dim = 4096 # feature dimensions of each regions
project_dim = 1024 # project the features to one vector, which is 1024 dimensions

sentRNN_lstm_dim = 1024 # the sentence LSTM hidden units
sentRNN_FC_dim = 1024 # the fully connected units
wordRNN_lstm_dim = 512 # the word LSTM hidden units
word_embed_dim = 1024 # the learned embedding vectors for the words

S_max = 6
N_max = 10
T_stop = 0.5


#500
n_epochs = 1
learning_rate = 0.001


# In[3]:


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
        self.n_words = n_words
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
            #sentRNN_binary = sentRNN_binary
            h1 = sent_topic_vec[:,:,0:512].transpose_(0,1)
            c1 = sent_topic_vec[:,:,512:].transpose_(0,1)
            h1= torch.cat((h1,h1),0)
            c1= torch.cat((c1,c1),0)
            #[000001111]
            temp_distribution = torch.cat((temp_distribution,sentRNN_binary),1)
            
            ############not sure 
            temp_output=Variable(torch.Tensor().long()).cuda()
#             for j in range(0, self.N_max):
                
#                 current_embed = self.embed(captions[:,i,j])
#                 print current_embed.size()
                
#                 current_embed=current_embed.unsqueeze(1)
#                 word_output,word_state = self.word_LSTM(current_embed,(h1,c1))
#                 word_output = self.m(self.word_FC(word_output))
#                 temp_output = torch.cat((temp_output,word_output),1)

            current_embed = self.embed(captions[:,i,:].long().cuda())
            #print(current_embed.size())
        
            #need to do pack padded sequence. 
            if i==0:
                word_output,word_state = self.word_LSTM(current_embed,(h1,c1))
            else:
                word_output,word_state = self.word_LSTM(current_embed,word_state)
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
    
    
        features = features.view(self.batch_size ,1,-1).LongTensor.cuda()
        temp_distribution= Variable(torch.Tensor()).cuda()
        final_output = Variable(torch.Tensor()).cuda()
        for i in range(0,self.S_max):
            
            sent_output,sent_state = self.sentLSTM(features)
            hidden1 = F.relu(self.sent_FC(sent_output))
            sent_topic_vec = F.relu(hidden1.LongTensor())
            
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
                _, predicted = word_output.max(1) 
                
                temp_output.append(predicted)
                c_embed = self.embed(predicted)
                #inputs = inputs.unsqueeze(1) 
            
            temp_output= temp_output.unsqueeze(1)
            final_output = torch.cat((final_output,temp_output),1)
        #print final_output.size()
        return temp_distribution,final_output


# In[4]:



def getIdx():
    ixtoword = {}
    wordtoix = {}
    ingr_vocab = {}
    #Vocab.text is the pre-processed corpus of all the instructions.
    with open('../files/vocab.txt') as f_vocab:
        #wordtoix a dict mapped to the index values. Key is the word and value is the index value.
        #ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua
        #ingr_vocab['</i>'] = 1
        #print (ingr_vocab)
        for i, w in enumerate(f_vocab):
            word = w.rstrip()
            ixtoword[i+5] = word
            wordtoix[w.rstrip()] = i+5
        wordtoix['</i>'] = 4
        wordtoix['<bos>'] = 0
        wordtoix['<eos>'] = 1
        wordtoix['<pad>'] = 2
        wordtoix['<unk>'] = 3
        ixtoword[0] = '<bos>'
        ixtoword[1] = '<eos>'
        ixtoword[2] = '<pad>'
        ixtoword[3] = '<unk>'
        ixtoword[4] = '</i>'
    return wordtoix,ixtoword


# In[7]:



from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
    
word2idx,ixtoword = getIdx()

#model = image_ingredient().cuda()
model = EncoderCNN(1024).cuda()
#decoder=DecoderRNN(1024, 512, len(word2idx), 2).cuda()
decoder = HierRNN(n_words = len(word2idx),
                                          batch_size = BATCH_SIZE,
                                          num_boxes = num_boxes,
                                          feats_dim = feats_dim,
                                          project_dim = project_dim,
                                          sentRNN_lstm_dim = sentRNN_lstm_dim,
                                          sentRNN_FC_dim = sentRNN_FC_dim,
                                          wordRNN_lstm_dim = wordRNN_lstm_dim,
                                          S_max = S_max,
                                          N_max = N_max,
                                          word_embed_dim = word_embed_dim
                                            #bias_init_vector = bias_init_vector
                     ).cuda()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(model.linear.parameters()) + list(model.bn.parameters())
optimizer = torch.optim.Adam(params,lr=learning_rate)

train_loader = torch.utils.data.DataLoader(
        ImageLoader(IMG_PATH,
            transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(256), # we get only the center of that rescaled
            transforms.RandomCrop(224), # random crop within the center crop
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),data_path=DATA_PATH,partition='train'),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True)
print ('Training loader prepared.')
totalloss=[]
total_step = len(train_loader)
for epoch in range(1):
        #for i, (images, captions, lengths) in enumerate(data_loader):
        for i, (input, target) in enumerate(train_loader):
            # Set mini-batch dataset
            images = Variable(input[0]).cuda()
            x = Variable(input[4][:,:,0:10]).cuda()
#             targets = pack_padded_sequence(x, x_len, batch_first=True)[0]
            
            # Forward, backward and optimize
            #encoder_output = model(Variable(input[0]).cuda(), Variable(input[1]).cuda(), Variable(input[2]).cuda())
            encoder_output = model(images).float()
#             encoder_output = encoder_output[torch.LongTensor(x_sort_idx)]
            temp_distribution,outputs, = decoder(encoder_output, x)
     
            loss = criterion(outputs.transpose_(1, 3).transpose_(2,3), input[4][:,0:6,1:11].long().cuda())
            decoder.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
#             out = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # (sequence, lengths)
#             out = out[0]
#             print out
#             out = out[x_unsort_idx]
            
            
           
            # Print log info
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, 100, i, total_step, loss.item(), np.exp(loss.item()))) 
                torch.save(decoder.state_dict(), os.path.join(
                    "simpleModel/", 'cddecoder-{}.ckpt'.format(epoch+1)))
                torch.save(model.state_dict(), os.path.join(
                    "simpleModel/", 'cdencoder-{}.ckpt'.format(epoch+1)))


# In[ ]:


transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])


# In[ ]:


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


# In[ ]:


image = load_image("testFiles/5.jpg", transform)


# In[ ]:


image_tensor = image.cuda()


# In[ ]:


model = model.eval()
feature = model(image_tensor)
decoder = decoder.eval()
sampled_ids = decoder.sample(feature)
sampled_ids = sampled_ids[0].cpu().numpy()


# In[ ]:


sampled_ids


# In[ ]:


_,idx2word = getIdx()
sampled_caption = []
for word_id in sampled_ids:
    word = idx2word[word_id]
    sampled_caption.append(word)
    if word == '<end>':
        break
sentence = ' '.join(sampled_caption)


# In[ ]:


sentence

