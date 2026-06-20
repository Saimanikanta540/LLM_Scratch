import re
from Tokenizers.SimpleTokenizerV1 import SimpleTokenizerV1
from Tokenizers.SimpleTokenizerV2 import SimpleTokenizerV2
from Tokenizers.BPE_Tokenizer import BPE_Tokenizer
from GPTDataSetV1 import create_dataloader_v1

with open("Data/verdict.txt","r",encoding="utf-8") as file:
    content=file.read()

# preprocessed = re.split(r'([,.?;_:\'"()!]|--|\s)',content)
# preprocessed=[item.strip() for item in preprocessed if item.strip()]

# all_words=sorted(set(preprocessed))
# vocabulary_size=len(all_words)
# vocabulary={token:integer for integer,token in enumerate(all_words) }

# #v1 tokenizer
# v1=SimpleTokenizerV1(vocabulary)

# text = """I had always thought Jack Gisburn rather a cheap genius--though a
# good fellow enough--so it was no great surprise to me to hear that,"""

# ids=v1.encode(text)

# print(ids)
# print(v1.decode(ids))

# #Adding Special Context Tokens
# all_words=sorted(set(preprocessed))
# all_words.extend(["<|endoftext|>","<|unk|>"])
# vocabulary={token:integer for integer,token in enumerate(all_words) }

# #v2 tokenizer
# v2=SimpleTokenizerV2(vocabulary)

# text1="Hello, do you like tea?"
# text2="In the sunlit teerances of the place"
# concattext=" <|endoftext|> ".join((text1,text2))

# ids2=v2.encode(concattext)

# print(ids2)
# print(v2.decode(ids2))

#BPE Tokenizer
# bpe_tokenizer = BPE_Tokenizer("gpt2")
# bpe_text="Hi this is a radom text. <|endoftext|> this is a random word dksaimanikantakjdafk."
# bpe_tokenids=bpe_tokenizer.encoder(bpe_text)
# print("BPE Encoding")
# print(bpe_tokenids[:20])
# print(bpe_tokenizer.decoder(bpe_tokenids))

#GPTDatasetV1
# tokenizer=BPE_Tokenizer("gpt2")
# context_size=256
# stride=128
# dataset=GPTDatasetV1(content,tokenizer,context_size,stride)

# print("Total Samples:", len(dataset))

# x, y = dataset.__getitem__(0)

# print("\nInput Tokens")
# print(x)

# print("\nTarget Tokens")
# print(y)

# print("\nDecoded Input")
# print(tokenizer.decoder(x.tolist()))

# print("\nDecoded Target")
# print(tokenizer.decoder(y.tolist()))

loader=create_dataloader_v1(content)

dataiter=iter(loader)
firstbatch=next(dataiter)
print("First Batch :",firstbatch)
secondbatch=next(dataiter)
print("Second Batch :",secondbatch)


