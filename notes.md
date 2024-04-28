# Self notes for some research papers list
protip, use  perplexity.ai as a resource scrapper


## Optimization algorithm
- [x] SGD - AdamW / AdamScheduleFree (cs231n) [see blog](https://akash5100.github.io/blog/2024/04/12/Optimization_techniques.html)
- [ ] Hessian Matrix (second order, BFGS, LBFGS etc)

## Development of language models
- [x] Transformer (Attention is all you need)
  - [x] Self A, Multi Head A, Cross A (Attention)
  - [ ] GPT (2018) [LLMs are multitask learners (Ilya)]
  - [ ] BERT (2018)
  - [ ] T5 (2019)
  - [ ] RoBERTa (2019)
  - [ ] Transformer-XL (2019)
  - [ ] Reformer (2020)
  - [ ] FlashAttention
  - [ ] Longformer (2020)
  - [ ] Conformers (2020)
  - [ ] ViT (2020)
  - [ ] PaLM (2022)
  - [ ] Galactia (2022)
  - [ ] Whisper (2022)
  - [ ] Persimmon (2023)
  - [ ] Fuyu (2023)
  - [ ] Mamba, S4, SSM (2023)
  - [ ] InfiniAtten (2024)
  - [ ] Grouped Query Attention
  - [ ] Sliding Window Attention

## Models
- [ ] RNN
- [ ] LSTM
- [ ] GRU
- [ ] Seq2Seq (Ilya, 2014)


## Vision
- [ ] CNN Casestudy: 
  - [ ] CNN - Le - Alex - ZF - VGG - GoogleNet (inception architecture)
- [ ] Visualizing CNN techniques
  - [ ] DeepDream
- [ ] Localization and Segmentation (cs231n)
- [ ] Fast Faster Fastest?? R? - CNN's 
- [ ] ResNet (residual and skip connection, research paper)
- [ ] CLIP-ResNet (read somewhere kinda interesting, mostprobably best ResNet till date? not sure)
- [ ] COCO dataset (train something) 


- [ ] Multi-modality
- [ ] Generating Sequence with RNN (Alex graves, 2013)
- [ ] https://openai.com/research/weak-to-strong-generalization
- [ ] https://openai.com/research/microscope
- [ ] https://openai.com/research/language-models-can-explain-neurons-in-language-models
- [ ] Mistral of Experts
- [ ] Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention


## Image Generation
- [ ] Pixel RNN (maybe if interested)
- [ ] VAE
- [ ] GAN
- [ ] Stable Diffusion

## Reinforcement learning
- [ ] RF
- [ ] RoPE (it goes here? dont know.)

## Normalization and Regularization
- [ ] RMSNormLayer # most used, I think nowadays
(TODO: add more recently used normalization layers in chronological order, to know how one improved other for eg)
- [ ] LayerNorm (research paper)

## Some research papers
- [x] Understanding deeplearning requires rethinking generalization (research paper)-- See mnist_generalization [notebook](./mnist_generalization.ipynb)
- [ ] LM are Few Shot learners (https://arxiv.org/pdf/2005.14165)
- [ ] The recurrent temporal restricted boltzmann machine (research paper)-- energy model sounds interesting!
- [ ] Faster Training: Super Convergence (research paper)
- [ ] Training method: CURRICULUM LEARNING FOR LANGUAGE MODELING (research paper)

## Some resourceful repos 
- [ ] https://github.com/coqui-ai/TTS?tab=readme-ov-file#model-implementations

## Tokenization
- [ ] https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- [ ] https://arxiv.org/pdf/2207.14255.pdf
- [ ] https://arxiv.org/pdf/2304.08467.pdf
- [ ] https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/

## New (saw somewhere in twitter)
- [ ] Byte-level tokenization
- [ ] LM without tokens