import tiktoken

class BPE_Tokenizer:
    def __init__(self, model_name):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def encoder(self, text):
        return self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})

    def decoder(self, tokens):
        return self.tokenizer.decode(tokens)
