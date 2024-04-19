#!/usr/bin/env python
import regex as re
from base import BaseTokenizer, merge, get_stats 


LLAMA3_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN   = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


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


if __name__ == '__main__':

  tcn = BasicTokenizer()
  # loading data to train tokenizer
  # with open('../data/cs.txt', 'r') as f:
  #   text = f.read()
  # tcn.train(text, 256 + 3000, verbose=True) # 256 are the byte tokens, then do 3k merges
  # tcn.save('bpe3k')

  tcn.load('bpe3k.model')
  tcn.encode('Computer Science, technology')
  # [1634, 962, 559, 121]
  tcn.decode([1634, 962, 559, 121])
  # 'Computer Science, technology'