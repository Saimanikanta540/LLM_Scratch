import re
from simpleTokenizerV1 import simpleTokenizerV1
with open("Data/verdict.txt","r",encoding="utf-8") as file:
    content=file.read()

preprocessed = re.split(r'([,.?;_:\'"()!]|--|\s)',content)
preprocessed=[item.strip() for item in preprocessed if item.strip()]

all_words=sorted(set(preprocessed))
vocabulary_size=len(all_words)

vocabulary={token:integer for integer,token in enumerate(all_words) }

v1=simpleTokenizerV1(vocabulary)
