# GPT
# BERT
# CREATE A TLOKENIZER
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}

    def tokenize(self, text):
        tokens = text.split()  # Simple whitespace tokenizer
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(idx, '[UNK]') for idx in ids]