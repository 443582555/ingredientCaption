from __future__ import print_function
from PIL import Image
import torch.utils.data as data
import os
import sys
import pickle
import numpy as np
import lmdb
import torch
import pdb

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        #print("...", file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        print('Here\'s the path {}'.format(path))
        #pdb.set_trace()
        return Image.new('RGB', (224, 224), 'white')

    
class ImageLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None):
        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition
        # Open the LMDB files.
        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)
        
        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 20
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def getimgIdx(self, imgs):

        img = Image.new('RGB', (224, 224), 'white')

        for imgIdx in range(len(imgs)):

            loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]

            loader_path = os.path.join(*loader_path)

            path = os.path.join(self.imgPath, self.partition, loader_path, imgs[imgIdx]['id'])

            img = self.loader(path)

            if img != Image.new('RGB', (224, 224), 'white'):

                break

        if img == Image.new('RGB', (224, 224), 'white'):

            print('no images for sample found')

            #pdb.set_trace()

        return imgIdx 
    
    def __getitem__(self, index):

        recipId = self.ids[index]

        # we force 80 percent of them to be a mismatch
        #Change this to train
        if self.partition == 'train':
            match = np.random.uniform() > self.mismtch
        elif self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise 'Partition name not well defined'

        #target = match and 1 or -1 
        target = 1
        #pdb.set_trace()
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode())
            if not serialized_sample:
                print("id {} not found in lmdb".format(self.ids[index]))

        #pdb.set_trace()
        sample = pickle.loads(serialized_sample)
        #for img in sample:
        #    if we have the image:
        #        new_sample.append(img)
        #pdb.set_trace()
        imgs = sample['imgs']
        #print(target)
        # image
        if target == 1:
            #Change this to train
            if self.partition == 'train':
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(imgs))))
                
            else:
                imgIdx = 0
            loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
            loader_path = os.path.join(*loader_path)
            path = os.path.join(self.imgPath, self.partition, loader_path, imgs[imgIdx]['id'])
        """    
        else:
            # we randomly pick one non-matching image
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            with self.env.begin(write=False) as txn:
                serialized_sample = txn.get(self.ids[rndindex].encode())

            rndsample = pickle.loads(serialized_sample)
            rndimgs = rndsample['imgs']
            
            #Change this to train
            if self.partition == 'train':  # if training we pick a random image
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(rndimgs))))
                #print ('heelooo')
            else:
                imgIdx = 0

            path = self.imgPath + rndimgs[imgIdx]['id']
        """
        #print ('heelooo')
      
        # instructions
        
        instrs = sample['instruct']
        instr_vec_sent = instrs[0]
        instr_vec_word = instrs[1]
        numSentence = instrs[2]
        numWords = instrs[3]
        img_caption_one_matrix=instrs[4]
        oneSentenceLength = instrs[5]
#         t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
#         t_inst[:itr_ln][:] = instrs
        instr_vec_sent = torch.FloatTensor(instr_vec_sent)
        instr_vec_word = torch.FloatTensor(instr_vec_word)
        img_caption_one_matrix = torch.FloatTensor(img_caption_one_matrix)
        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        # load image
        img = self.loader(path)

        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rec_class = sample['classes'] - 1
        rec_id = self.ids[index]

        if target == -1:
            img_class = rndsample['classes'] - 1
            img_id = self.ids[rndindex]
        else:
            img_class = sample['classes'] - 1
            img_id = self.ids[index]
        #Change this to train
        if self.partition == 'train':
            return [img, ingrs, igr_ln, instr_vec_sent, instr_vec_word,numSentence,numWords,img_caption_one_matrix,oneSentenceLength], [target]
            #return [img, instrs, itr_ln, ingrs, igr_ln], [target]
        
        else:
            return [img, ingrs, igr_ln , instr_vec_sent, instr_vec_word,numSentence,numWords,img_caption_one_matrix,oneSentenceLength], [target, img_id, rec_id]
            #return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_id, rec_id]

    def __len__(self):
        return len(self.ids)    
    
    