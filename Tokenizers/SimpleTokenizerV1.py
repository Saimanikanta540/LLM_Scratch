import re

class SimpleTokenizerV1:

    def __init__(self,vocabulary):
        self.encoderVocabulary=vocabulary
        self.decoderVocabulary={tokenid:token for token,tokenid in vocabulary.items()}

    def encode(self,text):
        preprocessed_input_text = re.split(r'([,.:;?!_()"\']|--|\s)',text)
        preprocessed_input_text = [
            item.strip() for item in preprocessed_input_text if item.strip()
        ]
        input_text_ids=[self.encoderVocabulary[token] for token in preprocessed_input_text]
        return input_text_ids
    
    def decode(self,input_text_ids):
        result_text=" ".join(self.decoderVocabulary[tokenid] for tokenid in input_text_ids)
        # Replace white spaces before specified symbols mentioned below
        result_text=re.sub(r'"\s+([,.:;?!_()"\'])',r'\1',result_text)    
        return result_text