import torch
from Tokenizers.BPE_Tokenizer import BPE_Tokenizer
from torch.utils.data import Dataset,DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,context_size,stride):
        self.input_ids=[]
        self.target_ids=[]

        token_ids=tokenizer.encoder(txt)

        for i in range(0,len(token_ids)-context_size,stride):
            input_chunk = token_ids[i:i+context_size]
            target_chunk= token_ids[i+1:i+context_size+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
    
def create_dataloader_v1(txt,batch_size=4,context_size=256,stride=128,shuffle=True,drop_last=True,num_workers=0):

    #intiales the tokenizer
    tokenizer=BPE_Tokenizer("gpt2")

    #creating dataset
    dataset = GPTDatasetV1(txt,tokenizer,context_size,stride)

    #creating dataloader
    dataloader= DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)

    return dataloader

