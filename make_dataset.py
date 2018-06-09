
import json
import numpy as np
import copy
import pickle
import lmdb

print('Loading Ingredients vocab.')
# Making the dict of the vocab of the ingredients.
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
       # print(w)

        
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

LAYER_1_PATH = '../layer1.json'
LAYER_2_PATH = '../layer2.json'
DET_INGRS_PATH = '../det_ingrs.json'
CLASSES1M_PKL_PATH = '../classes1M.pkl'
TRAIN_LMDB_PATH = '/home/yifu/Documents/Mycode/python/hierarchicalRNN/jasha/lmdb/train_lmdb'
VALID_LMDB_PATH = '/home/yifu/Documents/Mycode/python/hierarchicalRNN/jasha/lmdb/val_lmdb'
TEST_LMDB_PATH =  '/home/yifu/Documents/Mycode/python/hierarchicalRNN/jasha/lmdb/test_lmdb'

MAX_LENGTH_INGRS = 20
MAX_NUM_IMGS = 5

#Open the det_ingrs.json.
json_det_ingrs = open(DET_INGRS_PATH).read() 
# data set for all the ingredients. Store the content in dataset_det_ingrs
dataset_det_ingrs = json.loads(json_det_ingrs) 
print ("Loaded Det Ingrs")

#Open the layer1.json file
json_layer_1 = open(LAYER_1_PATH).read()
dataset_layer_1 = json.loads(json_layer_1) 
print ("Loaded Layer 1")

#Open the layer2.json file
json_layer_2 = open(LAYER_2_PATH).read()
dataset_layer_2 = json.loads(json_layer_2)
print ("Loaded Layer 2")



with open(CLASSES1M_PKL_PATH,'rb') as f:
    class_dict = pickle.load(f)
    id2class = pickle.load(f)
    
# Still in doubt how this function work, but basically it return the combined .json files.
def merge(layers):
    base = layers[0]
    entries_by_id = {entry['id']: entry for entry in base}
    for layer in layers[1:]:
        for entry in layer:
            base_entry = entries_by_id.get(entry['id'])
            if not base_entry:
                continue
            base_entry.update(entry)
    return base

# Merging the data set into one list. 
dataset = merge([dataset_layer_1,dataset_layer_2,dataset_det_ingrs])
def preprocessSencentes(listOfSentences,S_max,N_max,word2idx):
    numOfSentences = len(listOfSentences)
    numForEachSentences = []
    sentences=[]
    for i in listOfSentences:
        sentences.append(i['text'])
    if '' in sentences:
        sentences.remove('')
    if numOfSentences > S_max:
        numOfSentences = S_max  
    img_num_distribution = np.zeros([S_max], dtype=np.int32)
    img_num_distribution[numOfSentences-1:] = 1
    img_captions_matrix = np.ones([S_max, N_max+1], dtype=np.int32) * 2 # zeros([6, 50])
    img_caption_one_matrix = np.ones([100],dtype=np.int32)*2
    catSentence = ""
    for i in sentences:
        i = i.replace(',', ' ,')
        if i[0] == ' ' and i[1] != ' ':
            i = i[1:]
        elif i[0] == ' ' and i[1] == ' ' and i[2] != ' ':
            i = i[2:]

        if i[-1] == '.':
            i = i[0:-1]
        elif i[-1] == ' ' and i[-2] == '.':
            i = i[0:-2]
        catSentence +=  i
    catSentence = '<bos> ' + catSentence + ' <eos>'
    oneSentenceLength =0
    for idx, word in enumerate(catSentence.lower().split(' ')):
        oneSentenceLength+=1
        if idx == 100:
                break        # the number of sentences is img_num_sents
        if word in word2idx:
                img_caption_one_matrix[idx] = word2idx[word]
        else:
                img_caption_one_matrix[idx] = word2idx['<unk>']
        # because we treat the ',' as a word

    for idx, img_sent in enumerate(sentences):
        # Because I have preprocess the paragraph_v1.json file in VScode before,
        # and I delete all the 2, 3, 4...bankspaces
        # so, actually, the 'elif' code will never run
        if idx==S_max:
            break
        img_sent = img_sent.replace(',', ' ,')
        
        if img_sent[0] == ' ' and img_sent[1] != ' ':
            img_sent = img_sent[1:]
        elif img_sent[0] == ' ' and img_sent[1] == ' ' and img_sent[2] != ' ':
            img_sent = img_sent[2:]

        # Be careful the last part in a sentence, like this:
        # '...world.'
        # '...world. '
        if img_sent[-1] == '.':
            img_sent = img_sent[0:-1]
        elif img_sent[-1] == ' ' and img_sent[-2] == '.':
            img_sent = img_sent[0:-2]
            
        mlength=0
        
        # Last, we add the <bos> and the <eos> in each sentences
        img_sent = '<bos> ' + img_sent + ' <eos>'

        # translate each word in a sentence into the unique number in word2idx dict
        # when we meet the word which is not in the word2idx dict, we use the mark: <unk>
        for idy, word in enumerate(img_sent.lower().split(' ')):
            # because the biggest number of words in a sentence is N_max, here is 50
            if idy == N_max:
                break

            if word in word2idx:
                img_captions_matrix[idx, idy] = word2idx[word]
            else:
                img_captions_matrix[idx, idy] = word2idx['<unk>']
            mlength = mlength+1
        numForEachSentences.append(mlength)

    # Pay attention, the value type 'img_name' here is NUMBER, I change it to STRING type
    return [img_num_distribution, img_captions_matrix,numOfSentences,numForEachSentences,img_caption_one_matrix,oneSentenceLength]


    
keys = {'train' : [], 'val':[], 'test':[]}
env = {'train' : [], 'val':[], 'test':[]}
env['train'] = lmdb.open(TRAIN_LMDB_PATH ,map_size=int(1e11))
env['val']   = lmdb.open(VALID_LMDB_PATH ,map_size=int(1e11))
env['test']  = lmdb.open(TEST_LMDB_PATH ,map_size=int(1e11))

# This funtion gives the index value of the ingredients of a single recipe in a list. 
def detect_ingrs(recipe, vocab):
    try:
        ingr_names = [ingr['text'] for ingr in recipe['ingredients'] if ingr['text']]
    except:
        ingr_names = []
        print ("Could not load ingredients! Moving on...")

    detected = set()
    for name in ingr_names:
        name = name.replace(' ','_')
        name_ind = vocab.get(name)
        if name_ind:
            detected.add(name_ind)
        '''
        name_words = name.lower().split(' ')
        for i in xrange(len(name_words)):
            name_ind = vocab.get('_'.join(name_words[i:]))
            if name_ind:
                detected.add(name_ind)
                break
        '''
    return list(detected) + [vocab['</i>']]


count = 0

for i,entry in enumerate(dataset):
    
    # Get the list containg the index value of the ingredients from the vocab.text
    ingr_detections = detect_ingrs(entry, wordtoix)
    #Length of the ingredients in single recipe.
    length_ingrs = len(ingr_detections)
    
    imgs = entry.get('images')
    if imgs:
        count += 1
    if length_ingrs >= MAX_LENGTH_INGRS or length_ingrs == 0 or not imgs:
        continue
    #Make a constant length list. and store the content of the variable length ingredients list in it.
    # So that each ingredient has same length vec.
    ingr_vec = np.zeros((MAX_LENGTH_INGRS), dtype='uint16')
    ingr_vec[:length_ingrs] = ingr_detections
    img_num_distribution, img_captions_matrix,numOfSentences,numForEachSentences,img_caption_one_matrix,oneSentenceLength = preprocessSencentes(dataset[i]['instructions'],30,20,wordtoix)
    partition = entry['partition']
    
    serialized_sample = pickle.dumps( {'ingrs':ingr_vec,
        'classes':class_dict[entry['id']]+1, 'imgs':imgs[:MAX_NUM_IMGS], 'instruct':[img_num_distribution, img_captions_matrix,numOfSentences,numForEachSentences,img_caption_one_matrix,oneSentenceLength]} )
    
    with env[partition].begin(write=True) as txn:
        txn.put('{}'.format(entry['id']), serialized_sample)
    
    # keys to be saved in a pickle file
    keys[partition].append(entry['id'])
    #dummy += 1
    if i % 10 == 0:
        print ("{} This much is done".format(i))

print ("Done Creating the LMDB's")        
for k in keys.keys():
    with open('/home/yifu/Documents/Mycode/python/hierarchicalRNN/jasha/lmdb/{}_keys.pkl'.format(k),'wb') as f:
        pickle.dump(keys[k],f)
        
print('Training samples: %d - Validation samples: %d - Testing samples: %d' % (len(keys['train']),len(keys['val']),len(keys['test'])))    






