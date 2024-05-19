#!/usr/bin/env python
import regex as re
from .base import BaseTokenizer, merge, get_stats 


LLAMA3_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN   = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Special tokens
STARTOFTEXT = '<|startoftext|>'
ENDOFTEXT = '<|endoftext|>'


class BasicTokenizer(BaseTokenizer):
  def __init__(self):
    super().__init__()

  def train(self, text, vocab_size=256, verbose=False):
    """builds merges"""
    assert vocab_size >= 255
    num_merges = vocab_size - 256
    merges = {}
    vocab = {i:bytes([i]) for i in range(256)}
    tokens = list(text.encode('utf-8'))
    for i in range(num_merges):
      stats = get_stats(tokens)
      pair = max(stats, key=stats.get) # maximum occured pair
      idx = 256+i
      tokens = merge(tokens, pair, idx)
      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]] # instead of _build, create while running
      if verbose:
        print(f"[{pair[0]:5d}, {pair[1]:5d}] -> {256+i}")

    self.merges = merges
    self.vocab = vocab


  def encode(self, text: str) -> list:
    tokens = text.encode('utf-8')
    while len(tokens) > 1:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      tokens = merge(tokens, pair, idx) 
    return tokens


  def decode(self, tokens: list) -> str:
    bin_text = b"".join(self.vocab[i] for i in tokens)
    return bin_text.decode('utf-8')


class RegexTokenizer(BasicTokenizer):
  def __init__(self, special_tokens=None, pattern=LLAMA3_SPLIT_PATTERN):
    super().__init__()
    self.pattern = pattern
    self.compiled_pattern = re.compile(self.pattern)
    self.special_tokens = special_tokens if special_tokens else {}
    self.inverse_special_tokens = {}


  def train(self, text, vocab_size=256, verbose=False):
    """text -> regex -> builds merges"""
    assert vocab_size >= 255
    num_merges = vocab_size - 256
    merges = {}

    text_chunks = self.compiled_pattern.findall(text)
    tokens = [list(i.encode('utf-8')) for i in text_chunks]
    vocab = {i:bytes([i]) for i in range(256)}
    for i in range(num_merges):
      stats = {}
      for c in tokens:
        stats = get_stats(c, stats)
      pair = max(stats, key=stats.get) # maximum occured pair
      idx = 256+i
      tokens = [merge(chunk, pair, idx) for chunk in tokens]
      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]] # instead of _build, create while running
      if verbose:
        print(f"[{pair[0]:5d}, {pair[1]:5d}] -> {256+i}")
    self.merges = merges
    self.vocab = vocab


  def register_special_tokens(self) -> None:
    # { ENDOFTEXT: 3257 }
    self.inverse_special_tokens = {v:i for i,v in self.special_tokens.items()}

  def _encode_chunk(self, tokens: list) -> list:
    while len(tokens) > 1:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      tokens = merge(tokens, pair, idx) 
    return tokens


  def encode_ordinary(self, text: str) -> list:
    # ignores special tokens
    text_chunks = self.compiled_pattern.findall(text)
    tokens = []
    for ch in text_chunks:
      chunk = list(ch.encode('utf-8'))
      ids = self._encode_chunk(chunk)
      tokens.extend(ids)
    return tokens


  def encode(self, text: str) -> list:
    # handles special tokens
    special = self.special_tokens
    if len(special) == 0:
      return self.encode_ordinary(text)
    self.register_special_tokens()
    # we need to escape the special tokens
    special_pattern = '(' + '|'.join(re.escape(k) for k in special) + ')'
    special_chunks = re.split(special_pattern, text)
    tokens = []
    for chunk in special_chunks:
      if chunk in special:
        tokens.append(special[chunk])
      else:
        tokens.extend(self.encode_ordinary(chunk))
    return tokens


  def decode(self, tokens: list) -> str:
    ids = []
    for id in tokens:
      if id in self.vocab:
        ids.append(self.vocab[id])
      elif id in self.inverse_special_tokens:
        ids.append(self.inverse_special_tokens[id].encode('utf-8'))
      else:
        raise ValueError(f"invalid token id: {id}")
    text_bytes = b"".join(ids)
    return text_bytes.decode("utf-8", errors="replace")


if __name__ == '__main__':

  tcn = RegexTokenizer(special_tokens={
    STARTOFTEXT: 3256, # 3000 + 255 + 1'st # testing
    ENDOFTEXT: 3257,
  })
  # loading data to train tokenizer
  # with open('../data/cs.txt', 'r') as f:
  #   text = f.read()
  # tcn.train(text, 256 + 5000, verbose=True) # 256 are the byte tokens, then do 5k merges
  # tcn.save('bpe5k')

  tcn.load('bpe5k.model')
  tokens = tcn.encode_ordinary('hello world!!!? (안녕하세요!) lol123 😉')
  tcn.decode(tokens) == 'hello world!!!? (안녕하세요!) lol123 😉'
  # 'Computer Science, technology'

  assert 'Computer Science, technology' == tcn.decode(tcn.encode_ordinary('Computer Science, technology'))

  tokens = tcn.encode('<|startoftext|> hello world!!! <|endoftext|>')
  # print(tokens) # [3256, 337, 539, 111, 1700, 33, 33, 33, 32, 3257]
  # 3256 -> start of text 
  # 3257 -> end of text 
  assert tcn.decode(tokens) == '<|startoftext|> hello world!!! <|endoftext|>'

  text = 'In the ever-evolving landscape of computer science, advancements in artificial intelligence and machine learning continue to reshape industries and societies. From developing sophisticated algorithms for data analysis to creating intelligent systems capable of autonomous decision-making, the field of computer science is at the forefront of innovation. As researchers delve deeper into topics like neural networks, natural language processing, and computer vision, the possibilities seem endless. With each breakthrough, the boundaries of what computers can achieve expand, opening up new avenues for exploration and discovery. In this fast-paced digital age, staying abreast of the latest developments in computer science is essential'
  tokens = tcn.encode(text)
  assert tcn.decode(tokens) == text
  print(len(list(text.encode('utf-8'))), len(tokens)) # 736, 180
  print("compression ratio", round(100 - len(tokens) / len(list(text.encode('utf-8'))) * 100, 4), "%")
  # compression ratio 75.5435 %