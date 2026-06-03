import re
from SimpleTokenizerV1 import SimpleTokenizerV1
with open("Data/verdict.txt","r",encoding="utf-8") as file:
    content=file.read()

preprocessed = re.split(r'([,.?;_:\'"()!]|--|\s)',content)
preprocessed=[item.strip() for item in preprocessed if item.strip()]

all_words=sorted(set(preprocessed))
vocabulary_size=len(all_words)

vocabulary={token:integer for integer,token in enumerate(all_words) }

v1=SimpleTokenizerV1(vocabulary)

text = """I had always thought Jack Gisburn rather a cheap genius--though a
good fellow enough--so it was no great surprise to me to hear that,"""

ids=v1.encode(text)
print(ids)
print(v1.decode(ids))



