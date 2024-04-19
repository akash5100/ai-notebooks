# karpthy/minbpe ripoff 

def get_stats(tokens: list, toks=None) -> dict:
  toks = toks if toks else {}
  for pair in zip(tokens, tokens[1:]):
    toks[pair] = toks.get(pair, 0) + 1
  return toks

def merge(tokens: list, pair: tuple, idx: int) -> list:
  i = 0
  new_tokens = []
  while i < len(tokens):
    if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
      new_tokens.append(idx)
      i+=2
    else:
      new_tokens.append(tokens[i])
      i+=1
  return new_tokens

def replace_control_characters(s: str) -> str:
  # we don't want to print control characters
  # which distort the output (e.g. \n or much worse)
  # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
  # http://www.unicode.org/reports/tr44/#GC_Values_Table
  import unicodedata
  chars = []
  for ch in s:
    if unicodedata.category(ch)[0] != "C":
      chars.append(ch) # this character is ok
    else:
      chars.append(f"\\u{ord(ch):04x}") # escape
  return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# base tokenizer class
class BaseTokenizer:
  def __init__(self) -> None:
    self.merges = {}                 # (int, int) -> int
    self.pattern = ""                # str
    self.special_tokens = {}         # str -> int, e.g. {'<|endoftext|>': 100257}
    self.vocab = self._build_vocab() # int -> bytes


  def _build_vocab(self):
    # vocab is simply, with rules, derived from merges
    vocab = {i:bytes([i]) for i in range(256)}
    for (p0, p1), v in self.merges.items():
      vocab[v] = vocab[p0] + vocab[p1]
    for tok, v in self.special_tokens.items():
      vocab[v] = tok.encode('utf-8')
    return vocab

  def train(self, text, vocab_size=256, verbose=False):
    raise NotImplementedError
  
  def encode(self, text):
    raise NotImplementedError

  def decode(self, ids):
    raise NotImplementedError

  def save(self, file_name):
    """
    Saves two files: file.vocab and file.model
    This is inspired (but not equivalent to!) sentencepiece's model saving:
    - model file is the critical one, intended for load()
    - vocab file is just a pretty printed version for human inspection only
    """
    model_file = file_name + ".model"
    with open(model_file, 'w') as f:
      # write the version, pattern and merges, that's all that's needed
      f.write("minbpe v1\n")
      # pattern used
      f.write(f"{self.pattern}\n")
      # numbers of special tokens
      f.write(f"{len(self.special_tokens)}\n")
      # write special tokens
      for special, idx in self.special_tokens.items(): 
        f.write(f"{special} {idx}\n")
      # write merges
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")

    vocab_file = file_name + ".vocab"
    inverted_merges = {idx: pair for pair, idx in self.merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
      for idx, token in self.vocab.items():
        # note: many tokens may be partial utf-8 sequences
        # and cannot be decoded into valid strings. Here we're using
        # errors='replace' to replace them with the replacement char �.
        # this also means that we couldn't possibly use .vocab in load()
        # because decoding in this way is a lossy operation!
        s = render_token(token)
        # find the children of this token, if any
        if idx in inverted_merges:
          # if this token has children, render it nicely as a merge
          idx0, idx1 = inverted_merges[idx]
          s0 = render_token(self.vocab[idx0])
          s1 = render_token(self.vocab[idx1])
          f.write(f"[{s0}] [{s1}] -> [{s}] {idx}\n")
        else:
          # (this should just be the first 256 tokens, the bytes)
          f.write(f"[{s}] {idx}\n")


  def load(self, file):
    """Loads the file.model"""
    assert file.endswith(".model")
    merges = {}
    special_tokens = {}
    idx = 256
    with open(file, 'r', encoding="utf-8") as f:
      # confirm metadata
      version = f.readline().strip()
      assert version == "minbpe v1"
      self.pattern = f.readline().strip()
      # special tokens
      num_special = int(f.readline().strip())
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)
      # read the merges
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    self.merges = merges
    self.special_tokens = special_tokens
    self.vocab = self._build_vocab()
