import torch
import torch.nn as nn
import torch.nn.parallel
import torch.legacy as legacy
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb

ING_WORD2VEC_DIM = 300
ING_RNN_DIM = 300
ING_WORD2VEC_PATH = '/home/yifu/Documents/Mycode/python/hierarchicalRNN/jasha/files/vocab.bin'
IMAGE_MODEL = 'resNet50'
IMAGE_LAYER_FINAL = 2048
EMBD_DIM = 1024


class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()
    def forward(self,x,dim):
        y = torch.cat(x,dim)
        return y



class ingredient_RNN (nn.Module):
    def __init__(self):
        super(ingredient_RNN, self).__init__()
        
        self.irnn = nn.LSTM(input_size= ING_WORD2VEC_DIM, hidden_size=ING_RNN_DIM, bidirectional=True, batch_first=True)
        
        #Get the size of the Vocab.
        _, vec = torchwordemb.load_word2vec_bin(ING_WORD2VEC_PATH) # give the vector of size 300
        
        #Creating the Embedding Matrix and then copy the vectors from Google WORD2VEC model to the embedding variable
        self.embs = nn.Embedding(vec.size(0), ING_WORD2VEC_DIM, padding_idx=0) # not sure about the padding idx
        
        self.embs.weight.data.copy_(vec)
    
    def forward(self, x, seq_lengths):

        # X is the variable, seq_lengths is the length of the ingredients of single Recipe
        
        ################################
        # Following things need to be under stood properly.
        #################################
        # sort sequence according to the length
        x=self.embs(x)
        #print "Dim of Embedding is {}".format(x.size())
        

        sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)

        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x)
        
        sorted_inputs = x.gather(0, index_sorted_idx.long())

        #print "Sorted Inputs = {}".format(sorted_inputs)
        #print sorted_inputs.size()
        #print "The value of sorted_len is {}".format(sorted_inputs)
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)

        #print "The value of packed_seq is {}".format(packed_seq) 
        #print "packed_seq = {}".format(packed_seq)
        #print "---------------------------------------------------------------------------1"    
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)


        #print ("out = {}".format(out))
        #print ("hidden = {}".format(hidden))
        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output = output.view(output.size(0),output.size(1)*output.size(2))
        #print "output = {}".format(output)
        return output

# Following thing need to be understood completely 
def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

class image_ingredient(nn.Module):
    def __init__(self):
        super(image_ingredient, self).__init__()
        if IMAGE_MODEL =='resNet50':

            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

            self.visual_embedding = nn.Sequential(
                nn.Linear(IMAGE_LAYER_FINAL, EMBD_DIM),
                nn.Tanh(),
            )

            self.ingredient_embedding = nn.Sequential(
                #ING_RNN_DIM is doubtful here.  If it needs to be multiplied or not.
                nn.Linear(ING_RNN_DIM*2, EMBD_DIM, EMBD_DIM),
                nn.Tanh(),
            )
            self.output_embedding = nn.Linear(2048, EMBD_DIM, EMBD_DIM)
                

        else:
            raise Exception('Only resNet50 model is implemented.')

        self.ingRNN_    = ingredient_RNN()
        self.table = TableModule()

    def forward(self, x, z1, z2): # we need to check how the input is going to be provided to the model
        
        #############################
        # X is the image matrix.
        # Change the input according, z1 and z2 retains the last two parameters for the ing_vec and sequence length
        ####################################
       
        # Ingredient embedding
        ingredient_emb = self.ingRNN_(z1,z2)
        ingredient_emb = self.ingredient_embedding(ingredient_emb)
        
        ingredient_emb = norm(ingredient_emb)
        
        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)
        output = torch.cat((visual_emb,ingredient_emb),1)
        output = self.output_embedding(output)
        return output
