#!/usr/bin/env python

import regex as re

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42); # reproducibility

# visualize hidden states, if we put layernorm before attention and after attention (GPT-1 vs GPT-2 architecture)
# Try to visualize Attention, for each context being generated.

# byte pair encoding implementation 
# https://en.wikipedia.org/wiki/Byte_pair_encoding

SL = 16

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

    #viz
    weights = torch.bmm(q, k.transpose(-2,-1)) * (head_sz**-0.5)
    weights = F.softmax(weights, dim=-1)

    return out, weights


## Multi-Head Attention
class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_sz):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_sz) for _ in range(n_head)])
    self.linear = nn.Linear(n_head*head_sz, emb_sz)
    self.dropout = nn.Dropout(P)
  
  def forward(self, x): # B,T,C
    out = []
    attention = []
    for h in self.heads:
      o, weights = h(x)
      out.append(o)
      attention.append(weights)
    out = torch.cat(out, dim=-1) # n* B,T,H -> B,T,n*H
    out = self.dropout(self.linear(out)) # B,T,nH @ nH, C -> B,T,C
    # viz
    attention = torch.stack(attention, dim=1)
    return out, attention


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
    self.attention = None # viz

  def forward(self, x): # B, T, C
    h, attention = self.sa(self.ln1(x))
    self.attention = attention # save to viz
    x = x + h
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
    if type(idx) is str:
      idx = torch.tensor(tkn.encode(idx), dtype=torch.long).unsqueeze(0)

    attentions = []
    for _ in range(max_new_tokens):
      i = idx[:, -SL:]
      logits, _ = self(i)

      # extra
      block_attention = []
      for block in self.blocks:
        block_attention.append(block.attention.squeeze(0))
      attentions.append(torch.stack(block_attention, dim=0))
      ####

      logits = logits[:,-1,:]
      probs = logits.softmax(-1)
      next_idx = torch.multinomial(probs, 1)
      idx = torch.cat((idx, next_idx), dim=-1)
      # yield next_idx
    return torch.stack(attentions, dim=0), idx.squeeze(0)

def get_batch(split, context_len=SL):
  data = train_data if split == 'train' else valid_data
  idx = torch.randint(0, len(data)-context_len, size=(bs,))
  x = torch.stack([data[i  :i+context_len  ] for i in idx])
  y = torch.stack([data[i+1:i+context_len+1] for i in idx])
  return x,y

@torch.no_grad()
def split_loss():
  eval_iters = 200  # run val per x data
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

def train(model, epoch, lr, warmup=False):
  optim = torch.optim.AdamW(model.parameters(), lr, (0.9, 0.995))
  if warmup:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch: min(1, (epoch + 1)/2000 ))
  interval = 100    # run val per x epoch
  save_interval = 500 # save every x epoch
  lossi, vlossi = [], []
  for i in range(epoch):
    xb,yb = get_batch('train')
    _, loss = model(xb, yb)
    # backward
    model.zero_grad(set_to_none=True)
    loss.backward()
    # step
    optim.step()
    if warmup:
      scheduler.step() #update lr
    # train-valid split testing (sometimes)
    if i % interval == 0 or i == epoch-1:
      tloss, vloss = split_loss()
      print(f"epoch {i:7d}/{epoch:7d} | loss: {tloss:7.4f} | perp: {(loss.exp().item()):7.4f} | vloss: {vloss:7.4f} | vperp: {(vloss.exp().item()):7.4f}")
    if i % save_interval == 0 or i == epoch-1:
      if len(vlossi) > 0 and vloss == min(vlossi):
        torch.save(model.state_dict(), 'gpt2.best.pth')
        print('==')
      torch.save(model.state_dict(), 'gpt2.1.pth')
      print("--")
    # track
    lossi.append(loss.item())
    vlossi.append(vloss.item())
  return lossi, vlossi

# Visualization helpers



if __name__ == '__main__':
  # creation of encoder and decoder
  # with open('./data/new.txt', 'r') as f:
  #   txt = f.read()
  # tokens = list(txt.encode('utf-8'))

  tkn = BPETokenizer()
  # # Hyperparameter
  # merges = 2000 - 256
  # tkn.train(txt, merges, True)
  # tkn.save(f'bpe2k')
  tkn.load(f'bpe2k.model')
  # text = 'hello world is my name??' # finetune and add trailing space infront
  # assert text == tkn.decode(tkn.encode(text))

  # =========== model init ===========
  ## GPT-2 Hyperparams
  bs = 32
  lr = 2.5e-4
  epoch = 2000

  # SL = 16
  vocab_sz = len(tkn.vocab) # 2k merges
  emb_sz = 64
  pos_sz = 64

  head_sz = 256
  n_head = 4
  n_layers = 5 # layers of stacked transformers

  # regularization
  P = 0.3 # dropout
  WD = 0.01 # l2 # not implemented (yet)

  # ====== dataset creation =========
  with open('./data/taylorswift.txt', 'r') as f: 
    text = f.read()
  tokens = torch.tensor(tkn.encode(text))
  n = int(len(tokens) * 0.8)
  train_data = tokens[:n]
  valid_data = tokens[n:]

  model = GPT2()
  model.load_state_dict(torch.load('./gpt2.1.pth'))

  print("==============================")
  print(str(round(sum([p.nelement() for p in model.parameters()]) / 1e6, 2)) + ' million parameters')
  print(str(sum([p.nelement() for p in model.parameters()])) + ' exactly')
  print("==============================")
  
  # ----- train -------
  # train(model, 1, 2.5e-4, warmup=False)
  # ----- ----- -------
  
  # sampling
  gen_tokens = 10
  prompt = " Join the Wiki­data contest and help improve geo­graphi­cally located items in 16 countries!"
  # print(prompt, end='')
  # # idx = torch.zeros((1, 1), dtype=torch.long)
  attentions, gen = model.generate(prompt, max_new_tokens=gen_tokens)
  for token in gen:
    token = [token.item()]
    print(tkn.decode(token), end='', flush=True)

  print(attentions.shape) # as expected: ([8, 5, 1, 4, 16, 16])
  torch.save(attentions, 'attentions.pt')
