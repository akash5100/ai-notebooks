# research on research papers to research stuffs for fun as a hobby
protip, use  perplexity.ai as a resource scrapper


## Optimization algorithm
- [x] SGD -> Momentum -> NAG -> AdaGrad -> RMSProp -> Adam -> AdamW -> AdamScheduleFree (2024, by FAIR) [read blog](https://akash5100.github.io/blog/2024/04/12/Optimization_techniques.html)
  Available in cs231n lecture.
- [ ] Hessian Matrix (second order, BFGS, LBFGS etc)
- [ ] AdamP, RAdam, and Stochastic Gradient Descent with Warm Restarts (SGDR)



## Development of language models
- [x] Transformer (Attention is all you need) [read blog](https://akash5100.github.io/blog/2024/04/28/Transformers.html)
  - [x] {Self, Multi Head, Cross} Attention
  - [x] GPT-1 (2018) / GPT-2 [GPT paper, LLMs are multitask learners] [read blog](https://akash5100.github.io/blog/2024/05/04/I_challenged_myself_to_visualize_attentions.html)
    - Summarization, still has some errors, didnt find exact fix to that problem but [this paper](https://arxiv.org/pdf/2305.04853) might have answer
  - [ ] BERT (2018)
  - [ ] RoBERTa (2019)
  - [ ] DistilBERT, ALBERT
  - [ ] T5 (2019)
  - [ ] TransformerXL (2019)
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
  - [ ] Mamba, S4, SSM (2023)  <--- does this belong here, yet to find
  - [ ] InfiniAtten (2024)
  - [ ] Grouped Query Attention
  - [ ] Sliding Window Attention
  - [ ] GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM


## Language Models
- [ ] RNN 'done update'
- [ ] LSTM 'done update'
- [ ] GRU 'done update'
- [ ] Seq2Seq (Ilya, 2014)
- [ ] JukeBox
- [ ] Mixture of experts (MoE)
- [ ] LLaMA
- [ ] Switch Transformers
- [ ] Multi-modality
- [ ] Beam Search?
- [ ] RAG
- [ ] Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention
- [ ] Greedy search used in LLMs for better predictions
- [ ] https://openai.com/index/language-models-can-explain-neurons-in-language-models


## Tokenization
- [ ] sentinel token [The procedure used in Donahue et al 2020, Aghajanyan et al., 2022, Fried et al., 2022]
- [x] https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- [ ] https://arxiv.org/pdf/2207.14255.pdf
- [ ] https://arxiv.org/pdf/2304.08467.pdf
- [x] https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/
  - "tokenization inconsistency" or "numerical tokenization problem"
  - Researchers have addressed this challenge in various ways:
    - Specialized tokenizers
    - MathBERT
    - Subword tokenization
      Google's PaLM (2022) use subword tokenization, which breaks down numbers into subwords (e.g., 123 → [1, 2, 3]). This allows for more consistent tokenization and better generalization to unseen numbers
    - Positional encoding: https://arxiv.org/pdf/2211.00170
- Math using tokens
  - GPT-2 -> Minerva 62B -> GPT-4 :)
  - WizardMath



## Vision
- [ ] CNN Casestudy: 
  - [ ] CNN - {Le - Alex - ZF - VGG - GoogLe}Net (inception architecture)
- [ ] Visualizing CNN techniques
  - [ ] DeepDream?
- [ ] Localization and Segmentation (cs231n)
- [ ] Fast Faster Fastest?? R? - CNN's 
- [ ] ResNet (residual and skip connection, research paper)
- [ ] CLIP-ResNet (read somewhere kinda interesting, mostprobably best ResNet till date? not sure)
- [ ] COCO dataset (train something) 
- [ ] yolo
- [ ] SSD?



## Image Generation
- [ ] Pixel RNN (maybe if interested)
- [ ] VAE
- [ ] GAN
- [ ] Stable Diffusion



## Reinforcement learning
- [ ] RF
- [ ] RoPE (it goes here? dont know.)
- [ ] DQN
- [ ] Policy Gradient Methods
- [ ] DPO



## Normalization and Regularization
- [ ] Quantization? Factorization?
- [ ] Weight Standardization
- [ ] Label Smoothing
- [ ] Filter Response Normalization 
- [ ] RMSNormLayer # most used, I think nowadays
(TODO: add more recently used normalization layers in chronological order, to know how one improved other for eg)
- [ ] LayerNorm (research paper)



## Some research papers
- [x] Understanding deeplearning requires rethinking generalization (research paper)-- See mnist_generalization [notebook](./mnist_generalization.ipynb)
- [ ] LM are Few Shot learners (https://arxiv.org/pdf/2005.14165)
- [ ] The recurrent temporal restricted boltzmann machine (research paper)-- energy model sounds interesting!
- [ ] Faster Training: Super Convergence (research paper)
- [ ] Training method: CURRICULUM LEARNING FOR LANGUAGE MODELING (research paper)
- [ ] https://openai.com/research/weak-to-strong-generalization
- [ ] https://openai.com/research/microscope
- [ ] https://openai.com/research/language-models-can-explain-neurons-in-language-models
- [ ] Generating Sequence with RNN (Alex graves, 2013)



## New (saw somewhere in twitter)
- [ ] Byte-level tokenization
- [ ] LM without tokens
- [ ] Kolmogorov–Arnold Networks (KANs-- learned activations performs better than MLPs)
- [ ] MambaByte--  token-free SSM for modeling long byte-sequences  [2401.13660](https://arxiv.org/pdf/2401.13660)
- [ ] Multi token Prediction-- https://arxiv.org/pdf/2404.19737



## Flops (Floating-Point Operations Per Second)
- [ ] gemm in C + Python (for noobs, use time monotonic!)



## Open Architectures
- [ ] Mixtral
- [ ] is Phi architecture open source?



## Some resourceful repos
- [ ] https://github.com/coqui-ai/TTS?tab=readme-ov-file#model-implementations