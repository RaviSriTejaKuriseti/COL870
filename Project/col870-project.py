import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import string
from string import digits
import re

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter

import random
from typing import Tuple
import torch.optim as optim
from torch import Tensor

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import log_softmax
import itertools



print(os.listdir("../input"))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)




lines=pd.read_csv("/kaggle/input/hindienglish-corpora/Hindi_English_Truncated_Corpus.csv",encoding='utf-8')
lines=lines[~pd.isnull(lines['english_sentence'])]
lines.drop_duplicates(inplace=True)
lines=lines[:10000]
lines=lines.sample(n=10000,random_state=42)

lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.lower())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.lower())
exclude = set(string.punctuation)
lines['english_sentence']=lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))


remove_digits = str.maketrans('', '', digits)
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))





lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.strip())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.strip())
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))





lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x : 'START_ '+ x + ' _END')


all_eng_words=set()
for eng in lines['english_sentence']:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_hindi_words=set()
for hin in lines['hindi_sentence']:
    for word in hin.split():
        if word not in all_hindi_words:
            all_hindi_words.add(word)



lines['length_eng_sentence']=lines['english_sentence'].apply(lambda x:len(x.split(" ")))
lines['length_hin_sentence']=lines['hindi_sentence'].apply(lambda x:len(x.split(" ")))



lines=lines[lines['length_eng_sentence']<=20]
lines=lines[lines['length_hin_sentence']<=20]

print("maximum length of Hindi Sentence ",max(lines['length_hin_sentence']))
print("maximum length of English Sentence ",max(lines['length_eng_sentence']))
max_length_src=max(lines['length_hin_sentence'])
max_length_tar=max(lines['length_eng_sentence'])





input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_hindi_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_hindi_words)
num_encoder_tokens, num_decoder_tokens
num_decoder_tokens += 1 #for zero padding





input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())





lines = shuffle(lines)



from torch.utils.data import Dataset

class NLP_Dataset(Dataset):
    
    def __init__(self,dataframe,max_length_src,max_length_tar,num_decoder_tokens):
        
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        
        source_text = self.dataframe.iloc[index].at["english_sentence"]
        target_text = self.dataframe.iloc[index].at["hindi_sentence"]
        
        S=np.zeros(max_length_src,dtype='int32')
        T=np.zeros((max_length_tar,num_decoder_tokens),dtype='float32')
        l=0
        
        I=source_text.split()
        l=len(I)
        
        for i,word in enumerate(I):
            S[i] = input_token_index[word]
            
        J=target_text.split()
        
        for t,word in enumerate(J):
             if t>0:
                T[t- 1,target_token_index[word]] = 1
                
        return torch.tensor(S),torch.tensor(l),torch.tensor(T)





from torch.utils.data import DataLoader
BATCH_SIZE=4





dataset = NLP_Dataset(lines,max_length_src,max_length_tar,num_decoder_tokens)
batch_size = 4
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=valid_sampler)



class Encoder(nn.Module):
    
    def __init__(self,vocab_size,num_layers=4,hidden_dim=128,dropout=0,batch_first=True,embedder=None,init_weight=0.4):
        super(Encoder,self).__init__()
        self.encoder_layers=nn.ModuleList()
        
        self.first_layer=nn.LSTM(hidden_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.second_layer=nn.LSTM(2*hidden_dim,hidden_dim,batch_first=True,bidirectional=False)
        
        self.encoder_layers.append(self.first_layer)
        self.encoder_layers.append(self.second_layer)
        self.dropout=nn.Dropout(dropout)
        
        if embedder is not None:
            self.embedding = embedder
        else:
            self.embedding = nn.Embedding(vocab_size,hidden_dim,padding_idx=0)
            nn.init.uniform_(self.embedding.weight.data,-init_weight,init_weight)
            
            
        
        for i in range(2,num_layers):
            self.encoder_layers.append(nn.LSTM(hidden_dim,hidden_dim,batch_first=True,bidirectional=False))
        
    
    def forward(self,input,lengths):
        
        #Input to Embedding
        x=self.embedding(input)

        # bidirectional layer
#         x = pack_padded_sequence(x, lengths.cpu().numpy(),batch_first=True)
        x, _ = self.encoder_layers[0](x)
#         x, _ = pad_packed_sequence(x, batch_first=True)
        
        
        #First unidirectional layer
        x=self.dropout(x)
        x,_=self.encoder_layers[1](x)
        
        #Second unidirectional layer onwards
        for i in range(2,len(self.encoder_layers)):
            residual=x
            x=self.dropout(x)
            output,_=self.encoder_layers[i](x)
            x=residual+output
        
#         print(x.shape)
        return x
        



class Attention(nn.Module):
    
    def __init__(self,query_dim,key_dim,hidden_dim,normalize,batch_first=True,init_weight=0.04):  
        '''
        Query comes from Decoder
        Key comes from Encoder Output
        '''
        super(Attention,self).__init__()
        
        self.Query=nn.Linear(query_dim,hidden_dim,bias=False)
        self.Key=nn.Linear(key_dim,hidden_dim,bias=False)
        self.hidden_dim=hidden_dim
        self.attention = Parameter(torch.Tensor(hidden_dim))
        self.mask = None
        self.batch_first=batch_first
        self.normalize=normalize
        
        std_dev=1.0/math.sqrt(hidden_dim)
        
        nn.init.uniform_(self.Query.weight.data, -1*init_weight, init_weight)
        nn.init.uniform_(self.Key.weight.data, -1*init_weight, init_weight)
        self.attention.data.data.uniform_(-1*init_weight, init_weight)
        
        if(self.normalize):
            self.scalar_constant=Parameter(torch.Tensor(1))
            self.scalar_constant.data.fill_(std_dev)
            self.bias_constant=Parameter(torch.Tensor(hidden_dim))
            self.bias_constant.data.zero_()
            
        else:
            self.register_parameter('scalar_constant',None)
            self.register_parameter('bias_constant',None)
            
            
    
    def set_mask(self,context_len,context):
        
        '''       
        b=context_len
        
        '''
        
        if self.batch_first:
            max_len = context.size(1)
        else:
            max_len = context.size(0)

        indices = torch.arange(0, max_len, dtype=torch.int64,device=context.device)
        self.mask = indices >= (context_len.unsqueeze(1))
        
    
    def score_calculation(self,query,key):
        
        """
        query=b*query_dim*n
        key=b*key_dim*n 
        
        """
        
        b,key_dim,n=key.size()
        query_dim=query.size(1)
        
        query=query.unsqueeze(2).expand(b,query_dim,key_dim,n)
        key=key.unsqueeze(1).expand(b,query_dim,key_dim,n)
        qk_sum=query+key
        
        linear_attention=None
        
        if(self.normalize):
            
            qk_sum=qk_sum+self.bias_constant
            linear_attention=(self.attention)/self.attention.norm()
            linear_attention=linear_attention*self.scalar_constant
            
            
        else:
            linear_attention=self.attention
            
        
        result=torch.tanh(qk_sum).matmul(linear_attention)
        return result
    
    
    def forward(self,Q,K):
        
        if (not self.batch_first):
            K=K.transpose(0,1)
            if(Q.dim()==3):
                Q=Q.transpose(0,1)
                
        if(Q.dim()==2):
            single_query=True
            Q=Q.unsqueeze()
        else:
            single_query=False
        
        b=Q.size(0)
        K_s=K.size(1)
        Q_s=Q.size(1)
        
        Q=self.Query(Q)
        K=self.Key(K)
        
        scores=self.score_calculation(Q,K)
        
        if(self.mask is not None):
            mask=self.mask.unsqueeze(1).expand(b,Q_s,K_s)
            scores.masked_fill_(mask,-65504.0)
        
        Normalized_scores=F.softmax(scores,dim=-1)
        Context=torch.bmm(Normalized_scores,K)
        
       
            
        if(single_query==True):
            
            Context=Context.squeeze(1)
            Normalized_scores=Normalized_scores.squeeze(1)
            
        elif(not self.batch_first):
            
            Context=Context.tranpose(0,1)
            Normalized_scores=Normalized_scores.transpose(0,1)
            
            
        
        return Context,Normalized_scores      
        





class Decoder(nn.Module):
    
    def __init__(self,vocab_size,num_layers=4,hidden_dim=128,dropout=0,batch_first=True,embedder=None,init_weight=0.4,attention_dropout=0):
        
        super(Decoder,self).__init__()
        
        self.num_layers=num_layers
        self.decoder_layers=nn.ModuleList()
        
        self.first_layer=nn.LSTM(vocab_size,hidden_dim,batch_first=True,bidirectional=False)
        
        self.attention_layer=Attention(hidden_dim,hidden_dim,hidden_dim,normalize=True,batch_first=True)
        self.Attention_Dropout=nn.Dropout(attention_dropout)
        
        self.dropout=nn.Dropout(dropout)
        
        if embedder is not None:
            self.embedding = embedder
        else:
            self.embedding = nn.Embedding(vocab_size,hidden_dim,padding_idx=0)
            nn.init.uniform_(self.embedding.weight.data,-init_weight,init_weight)
        
        
        self.classifier = nn.Linear(hidden_dim,vocab_size)
        self.H=[]
        
#         self.decoder_layers.append(self.first_layer)
#         self.decoder_layers.append(self.attention_layer)
        
        for i in range(1,num_layers):
            self.decoder_layers.append(nn.LSTM(2*hidden_dim,hidden_dim,batch_first=True,bidirectional=False))
            

        
        
    def forward(self,target,context,inference=False):
        
        self.inference=inference
        
        enc_context, enc_len, hidden = context
        
        if hidden is not None:
            hidden = hidden.chunk(self.num_layers)
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * self.num_layers
                
        
        x=target

        
        self.attention_layer.set_mask(enc_len,enc_context)
        
#         print(x.shape)
        
        x,h=self.first_layer(x,hidden[0])
        
       
        
        attn,scores=self.attention_layer(x,enc_context)
        x=self.Attention_Dropout(x)
        
        
        if(self.inference):
            (self.H).append(h)
            
        
        x = torch.cat((x, attn), dim=2)
        x = self.dropout(x)
        x, h = self.decoder_layers[0](x, hidden[1])
        
        if(self.inference):
            (self.H).append(h)
            
        for i in range(1, len(self.decoder_layers)):
            residual = x
            x = torch.cat((x, attn), dim=2)
            x = self.dropout(x)
            x, h = self.decoder_layers[i](x, hidden[i + 1])
            if(self.inference):
                (self.H).append(h)
            x = x + residual 
            
        x = self.classifier(x)

        if self.inference:
            hidden = torch.cat(tuple(itertools.chain(*self.H)))
        else:
            hidden = None
               
            
        return x, scores, [enc_context, enc_len, hidden]





class GNMT(nn.Module):
    
     def __init__(self,vocab_size,num_layers=4,hidden_dim=128,dropout=0,batch_first=True,share_embedding=False,init_weight=0.4,attention_dropout=0.1):
        
        super(GNMT,self).__init__()
        
        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_dim,0)
            nn.init.uniform_(embedder.weight.data,-0.4,0.4)
        else:
            embedder = None
            
        
        
        self.encoder=Encoder(vocab_size,num_layers,hidden_dim,dropout,batch_first,embedder,init_weight)
        self.decoder=Decoder(vocab_size,num_layers,hidden_dim,dropout,batch_first,embedder,init_weight,attention_dropout)
        
    
     def forward(self, input_encoder, input_enc_len, input_decoder,inference=False):
        
        context = self.encoder(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decoder(input_decoder,context,inference=False)
        return output




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=GNMT(num_decoder_tokens).to(device)



def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


import math
import time
from tqdm import tqdm


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src,l,trg) in enumerate(iterator):
        src,l,trg = src.to(device), l.to(device),trg.to(device)

        optimizer.zero_grad()
        
        output = model(src, l, trg[:, :-1])
        
        tgt_labels = trg[:, 1:]
        
        loss = criterion(output,tgt_labels)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)





def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src,l,trg) in enumerate(iterator):
            
            src,l,trg = src.to(device), l.to(device),trg.to(device)

            output = model(src, l, trg[:, :-1],True)
        
            tgt_labels = trg[:, 1:]

            loss = criterion(output,tgt_labels)            
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)





def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs





def complete_sentence(vec,dic):    
    s=""    
    while True:
        for e in vec:
            s1=dic[e]
            if(s1=="_END"):
                return s
            if(s1=="START_"):
                continue
            s+=s1
        return s
    return s







def generate(model,iterator,beam_size,reverse_target_char_index):
    
    with torch.no_grad():
        for _, (src,l,trg) in enumerate(iterator):
            src,l = src.to(device),l.to(device)            
            S=trg[:, :-1]
            input_decoder=torch.zeros_like(S).to(device) 
            logits= model(src,l,input_decoder,True)
            logprobs = log_softmax(logits, dim=-1)
            logprobs, words = logprobs.topk(beam_size, dim=-1)
            V=words.shape
            words=words.reshape(V[0],V[1]).tolist()
            for j in range(0,V[0]):
                print(complete_sentence(words[j],reverse_target_char_index))





N_EPOCHS = 20
CLIP = 1

training_loss=[]
validation_loss=[]

best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):

    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_loader, criterion)
    
    training_loss.append(train_loss)
    validation_loss.append(valid_loss)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_loader, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')



from matplotlib import pyplot as plt

plt.plot(training_loss)
plt.show()
plt.plot(validation_loss)


generate(model,test_loader,1,reverse_target_char_index)







