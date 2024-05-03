#!/usr/bin/env python

import regex as re

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42); # reproducibility

# ==> GPT-2 ==> Train & sampling
# visualize hidden states, if we put layernorm before attention and after attention (GPT-1 vs GPT-2 architecture)
# Try to visualize Attention, for each context being generated.

# byte pair encoding implementation 
# https://en.wikipedia.org/wiki/Byte_pair_encoding


# get the most occured pair in decending order
def get_stats(tokens, old_stats=None):
  toks = old_stats if old_stats else {}
  for pair in zip(tokens, tokens[1:]):
    toks[pair] = toks.get(pair, 0) + 1
  return toks

# merge the asked pair and replace it with idx
def merge(tokens, pair, idx):
  new = []
  i = 0
  while i<len(tokens):
    if i<len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
      new.append(idx)
      i+=2
    else:
      new.append(tokens[i])
      i+=1
  return new


# Regex used in the tokenizer
LLAMA3_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

class BPETokenizer:
  def __init__(self, pattern=LLAMA3_SPLIT_PATTERN):
    self.pattern = pattern
    self.compiled_pattern = re.compile(pattern)
    self.merges = {}
    self.vocab = self._build_vocab()

  # map for int->byte 
  def _build_vocab(self):
    vocab = {i:bytes([i]) for i in range(256)}
    for (p, idx) in self.merges.items():
      vocab[idx] = vocab[p[0]] + vocab[p[1]]
    return vocab

  # iteratively merges the most occuring pairs, we get compressed sequence(tokens)
  # text -> regex -> utf-8 -> builds merges
  def train(self, text, num_merges=256, verbose=False):
    merges = {}
    # 'hello world' ->['hello ', 'world ']
    text_chunks = self.compiled_pattern.findall(text) 
    tokens = [list(i.encode('utf-8')) for i in text_chunks]
    vocab = {i:bytes([i]) for i in range(256)}
    for i in range(num_merges):
      stats = {}
      for c in tokens:
        stats = get_stats(c, stats)
      pair = max(stats, key=stats.get)
      idx = 256 + i
      tokens = [merge(word, pair, idx) for word in tokens]
      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
      if verbose:
        print(i, "> " , pair, " --> ", idx)
    self.merges = merges
    self.vocab = vocab

  def _encode_chunk(self, tokens):
    while len(tokens) > 1:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      tokens = merge(tokens, pair, idx) 
    return tokens

  def _encode_ordinary(self, text: str) -> list:
    # keep merging untill the any pair is not in merge dict
    text_chunks = self.compiled_pattern.findall(text)
    tokens = []
    for ch in text_chunks:
      chunk = list(ch.encode('utf-8'))
      ids = self._encode_chunk(chunk)
      tokens.extend(ids)
    return tokens

  # text -> regex -> utf-8 -> merges -> token
  def encode(self, text: str) -> list:
    # deal with special tokens?
    return self._encode_ordinary(text)
  
  # token -> vocab(itos) -> text
  def decode(self, tokens):
    ids = []
    for id in tokens:
      if id in self.vocab:
        ids.append(self.vocab[id])
    toks = b''.join(ids)
    return toks.decode('utf-8', errors='replace')

  def save(self, file: str):
    model_file = file + '.model'
    with open(model_file, 'w') as f:
      f.write(f'{self.pattern}\n')
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")

  def load(self, file):
    assert file.endswith('.model')
    merges = {}
    with open(file, 'r', encoding='utf-8') as f:
      self.pattern = f.readline().strip()
      idx = 256
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    self.merges = merges
    self.vocab = self._build_vocab()

# Transformer's Core
## Self-Attention head (decoder)
class Head(nn.Module):
  def __init__(self, head_sz):
    super().__init__()
    self.query = nn.Linear(emb_sz, head_sz, bias=False)
    self.key   = nn.Linear(emb_sz, head_sz, bias=False)
    self.value = nn.Linear(emb_sz, head_sz, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(SL, SL)))
    self.dropout = nn.Dropout(P)
  
  def forward(self, x):
    B,T,C = x.shape
    q = self.query(x)
    k = self.key(x)
    wei = q @ k.transpose(-2,-1) * (head_sz**-0.5)
    wei = wei.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out


## Multi-Head Attention
class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_sz):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_sz) for _ in range(n_head)])
    self.linear = nn.Linear(n_head*head_sz, emb_sz)
    self.dropout = nn.Dropout(P)
  
  def forward(self, x): # B,T,C
    out = torch.cat([h(x) for h in self.heads], dim=-1) # n* B,T,H -> B,T,n*H
    out = self.dropout(self.linear(out)) # B,T,nH @ nH, C -> B,T,C 
    return out


## Feed Forward
class FeedForward(nn.Module):
  def __init__(self, emb_sz):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(emb_sz, emb_sz*4), # emb_sz*4 = 3072 (as mentioned in paper)
      nn.LayerNorm(emb_sz*4),
      nn.GELU(),
      nn.Linear(emb_sz*4, emb_sz),
      nn.Dropout(P),
    )
  
  def forward(self, x): # B,T,C
    out = self.net(x)
    return out

# GPT-2 style
class Transformer(nn.Module):
  def __init__(self, n_head, head_sz):
    super().__init__()
    assert head_sz%n_head == 0
    head_size = head_sz//n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ff = FeedForward(emb_sz)
    self.ln1 = nn.LayerNorm(head_size)
    self.ln2 = nn.LayerNorm(head_size)

  def forward(self, x): # B, T, C
    # self.ln1(x)
    x = x + self.sa(self.ln1(x))
    x = x + self.ff(self.ln2(x))
    return x
  
class GPT2(nn.Module):
  def __init__(self):
    super().__init__()
    self.vocab_emb = nn.Embedding(vocab_sz, emb_sz)
    self.positional_emb = nn.Embedding(SL, emb_sz)
    self.blocks = nn.Sequential(*[Transformer(n_head, head_sz) for _ in range(n_layers)])
    self.lnorm = nn.LayerNorm(emb_sz) # final layer norm
    self.linear = nn.Linear(emb_sz, vocab_sz)
    self.apply(self._init_weights)
 
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.constant_(module.bias, 0.001)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, x, target=None): # B,T
    B,T = x.shape
    tkn_emb = self.vocab_emb(x)
    pos_emb = self.positional_emb(torch.arange(T))
    x = tkn_emb + pos_emb
    x = self.blocks(x)
    x = self.lnorm(x)
    logits = self.linear(x)
    loss = None
    if target != None:
      B, T, C = logits.shape
      x = logits.view(B*T,C)
      y = target.view(B*T)
      loss = F.cross_entropy(x, y)
    return logits, loss
  
  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      i = idx[:, -SL:]
      logits, _ = self(i)
      logits = logits[:, -1, :] # B,T,C -> B,C
      probs = logits.softmax(-1)
      next_idx = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, next_idx), dim=1)
    return idx.squeeze()


if __name__ == '__main__':
  # Hyperparameter
  merges = 2000 - 256

  # creation of encoder and decoder
  # with open('./data/new.txt', 'r') as f:
  #   txt = f.read()
  # tokens = list(txt.encode('utf-8'))

  tkn = BPETokenizer()
  # tkn.train(txt, merges, True)
  # tkn.save(f'bpe2k')
  tkn.load(f'bpe2k.model')
  text = 'hello world is my name??' # finetune and add trailing space infront
  assert text == tkn.decode(tkn.encode(text))

  # model init ======================
  ## GPT-2 Hyperparams
  bs = 32
  lr = 2.5e-4
  # warmup = 2000 # warmup, linear increase in lr
  epoch = 2000 # converge

  SL = 8
  vocab_sz = len(tkn.vocab) # 2k merges
  emb_sz = 64
  pos_sz = 64

  head_sz = 256
  n_head = 4
  n_layers = 5 # layers of stacked transformers

  # regularization
  P = 0.1 # dropout
  WD = 0.01 # l2 # not implemented (yet)

  # ====== dataset creation =========
  with open('./data/taylorswift.txt', 'r') as f:
    text = f.read()
    f.close()
  tokens = tkn.encode(text)
  tokens = torch.tensor(tokens)
  n = int(len(tokens) * 0.9)
  train_data = tokens[:n]
  valid_data = tokens[n:]
  
  def get_batch(split, context_len=SL):
    data = train_data if split == 'train' else valid_data
    idx = torch.randint(0, len(data)-context_len, size=(bs,))
    x = torch.stack([data[i  :i+context_len  ] for i in idx])
    y = torch.stack([data[i+1:i+context_len+1] for i in idx])
    return x,y

  interval = 100    # run val per x epoch
  eval_iters = 200  # run val per x data
  lossi = []
  vlossi = []

  @torch.no_grad()
  def split_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        X, Y = get_batch(split)
        _, loss = model(X, Y)
        losses[k] = loss.item()
      out[split] = losses.mean()
    model.train()
    return out['train'], out['val']


  model = GPT2()
  optim = torch.optim.AdamW(model.parameters(), lr, (0.9, 0.995))

  print("==============================")
  print(str(round(sum([p.nelement() for p in model.parameters()]) / 1e6, 2)) + ' million parameters')
  print(str(sum([p.nelement() for p in model.parameters()])))
  print("==============================")

  # ----- train -------
  for i in range(epoch):
    # mini-batch
    xb,yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # backward
    optim.zero_grad(set_to_none=True)
    loss.backward()
    
    # step
    optim.step()
    
    # train-valid split testing (sometimes)
    if i % interval == 0 or i == epoch-1:
      tloss, vloss = split_loss()
      # print(f"epoch {i:7d}/{epoch:7d} | loss: {loss:7.4f} | perp: {(loss.exp().item()):7.4f}")
      print(f"epoch {i:7d}/{epoch:7d} | loss: {tloss:7.4f} | perp: {(loss.exp().item()):7.4f} | loss: {vloss:7.4f}")

    # track
    lossi.append(loss.item())
    vlossi.append(vloss.item())
  # ----- ----- -------
  
  torch.save(model.state_dict(), 'gpt2.pth')
  
  import matplotlib.pyplot as plt
  plt.plot(lossi) 
  plt.plot(vlossi) 
  plt.show()
  """
  ==============================
  0.76 million parameters
  755920
  ==============================
  epoch       0/   2000 | loss:  7.5541 | perp: 2012.4192 | loss:  7.5522
  epoch     100/   2000 | loss:  6.2247 | perp: 511.0773 | loss:  6.2492
  epoch     200/   2000 | loss:  5.6837 | perp: 271.5088 | loss:  5.7536
  epoch     300/   2000 | loss:  5.2621 | perp: 263.9460 | loss:  5.3231
  epoch     400/   2000 | loss:  4.8051 | perp: 100.9346 | loss:  4.9393
  epoch     500/   2000 | loss:  4.5254 | perp: 77.6216 | loss:  4.6859
  epoch     600/   2000 | loss:  4.2748 | perp: 63.5046 | loss:  4.5167
  epoch     700/   2000 | loss:  4.0867 | perp: 78.7867 | loss:  4.3576
  epoch     800/   2000 | loss:  3.9832 | perp: 73.2226 | loss:  4.2666
  epoch     900/   2000 | loss:  3.8379 | perp: 66.1574 | loss:  4.2452
  epoch    1000/   2000 | loss:  3.7920 | perp: 45.0464 | loss:  4.2208
  epoch    1100/   2000 | loss:  3.7132 | perp: 24.8325 | loss:  4.1533
  epoch    1200/   2000 | loss:  3.6406 | perp: 59.5014 | loss:  4.1499
  epoch    1300/   2000 | loss:  3.6134 | perp: 38.5099 | loss:  4.1112
  epoch    1400/   2000 | loss:  3.5232 | perp: 25.4662 | loss:  4.0760
  epoch    1500/   2000 | loss:  3.4874 | perp: 31.1807 | loss:  4.0921
  epoch    1600/   2000 | loss:  3.4362 | perp: 20.6329 | loss:  4.0134
  epoch    1700/   2000 | loss:  3.3913 | perp: 25.0268 | loss:  4.0013
  epoch    1800/   2000 | loss:  3.3134 | perp: 33.3010 | loss:  3.9572
  epoch    1900/   2000 | loss:  3.3111 | perp: 32.1315 | loss:  3.9889
  epoch    1999/   2000 | loss:  3.2927 | perp: 25.7719 | loss:  3.9600
  """