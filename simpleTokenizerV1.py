class SimpleTokenizerV1:
    def __init__(self,vocabulary):
        self.encoderVocabulary=vocabulary
        self.decoderVocabulary={tokenid:token for token,tokenid in vocabulary.items()}
    