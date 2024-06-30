# research on research papers to research stuffs for fun as a hobby
protip, use  perplexity.ai as a resource scrapper


## Optimization algorithm
- [x] SGD -> Momentum -> NAG -> AdaGrad -> RMSProp -> Adam -> AdamW -> AdamScheduleFree (2024, by FAIR) [read blog](https://akash5100.github.io/blog/2024/04/12/Optimization_techniques.html)
  - [ ] ScheduleFree, better than cosine decay -- https://arxiv.org/pdf/2405.15682 (Latest research on this part)
  Available in cs231n lecture.

## To learn later (not interested right now)
- [ ] Hessian Matrix (second order, BFGS, LBFGS etc)
- [ ] AdamP, RAdam, and Stochastic Gradient Descent with Warm Restarts (SGDR)
- [ ] Visualizing a loss landscape, interesting to implement (https://arxiv.org/abs/1712.09913)


## Development of transformer based models and architecture
- [x] Transformer (Attention is all you need) [blog](https://akash5100.github.io/blog/2024/04/28/Transformers.html)
  - [x] {Self, Multi Head, Cross} Attention
  - [ ] [Fast weights](https://arxiv.org/pdf/1610.06258) 
  - [x] GPT-1 (2018) / GPT-2 [GPT paper, LLMs are multitask learners] [blog](https://akash5100.github.io/blog/2024/05/04/I_challenged_myself_to_visualize_attentions.html)
    - Summarization, still has some errors, didnt find exact fix to that problem but [this paper](https://arxiv.org/pdf/2305.04853) might have answer
  - [x] BERT (2018) [blog](https://akash5100.github.io/blog/2024/05/09/Case_Study-_Transformer_based_architecture_development.html#bert-october-2018)
  - [x] TransformerXL (2019) [blog](https://akash5100.github.io/blog/2024/05/09/Case_Study-_Transformer_based_architecture_development.html#transformer-xl-september-2018-and-xlnet-june-2019)
  - [x] Sparse Transformer (2019)-- N sqrt(N) complexity. [blog](https://akash5100.github.io/blog/2024/05/09/Case_Study-_Transformer_based_architecture_development.html#sparse-transformers-april-2019)
  - [ ] RoBERTa, DistilBERT, ALBERT-- these are BERT variations, good to know
  - [ ] T5 (2019)-- Encoder-Decoder model
  - [ ] Reformer (2020)-- N log(N) complexity
  - [ ] Linformer-- linear complexity
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
- [x] [RNN](https://akash5100.github.io/blog/2024/02/06/Tweaking_MLP_to_make_it_RNN.html)
- [x] [LSTM](https://akash5100.github.io/blog/2024/02/07/LSTM.html)
- [x] [GRU](https://akash5100.github.io/blog/2024/02/16/GRU.html)
  - [ ] wait, should I try to train LSTM like I did for Transformers?
    - They dont support parallel computation, but recently **xLSTM** dropped which does.
- [ ] Seq2Seq (Ilya, 2014)
- [ ] JukeBox- openai
- [ ] Mixture of experts (MoE) [This?](https://arxiv.org/pdf/1701.06538)
- [ ] LLaMA- metaai
- [ ] Switch Transformers
- [ ] Multi-modality
  - [ ] https://arxiv.org/pdf/2405.09818v1
  - [ ] Vision-Language https://arxiv.org/abs/2405.17247 (May 2024)
- [ ] Beam Search?
- [ ] RAG
- [ ] Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention
- [ ] Greedy search used in LLMs for better predictions
- [ ] https://openai.com/index/language-models-can-explain-neurons-in-language-models
- [ ] Survey on Context length: https://arxiv.org/pdf/2402.02244

### Positional embeddings
- [ ] RoPE
- [ ] CoPE

## Tokenization
- [ ] sentinel token [The procedure used in *Donahue et al 2020, Aghajanyan et al., 2022, Fried et al., 2022*, described in Fill In Middle paper]
- [x] https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf (GPT-2 original paper)
- [x] https://arxiv.org/pdf/2207.14255.pdf (Fill in the middle tokens)
- [ ] https://arxiv.org/pdf/2304.08467.pdf (learning to compress prompts with GIST tokens)
- [x] https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/ (a blogs shows tokenization in LLMs is just pure memorization of artihmetics (next token prediction))
  - "tokenization inconsistency" or "numerical tokenization problem"
  - Researchers have addressed this challenge in various ways:
    - Specialized tokenizers
    - MathBERT
    - Subword tokenization
      Google's PaLM (2022) use subword tokenization, which breaks down numbers into subwords (e.g., 123 → [1, 2, 3]). This allows for more consistent tokenization and better generalization to unseen numbers
    - Positional encoding: https://arxiv.org/pdf/2211.00170 (math in transformer: Three results on interpretability and generalization)
- Math using tokens
  - GPT-2 -> Minerva 62B -> GPT-4 :)
  - WizardMath? 



## Finetuning
- [ ] LoRA: https://arxiv.org/abs/2106.09685



## Vision
- [x] CNN Casestudy: 
  - [x] CNN - { Le -> Alex -> ZF -> VGG -> Google }Net
    - TODO: (inception architecture)
- [x] ResNet (residual and skip connection, research paper)
- [x] Classification + Localization = Object detection (cs231n)
  - [x] R-CNN
  - [x] Fast R-CNN 
  - [x] Faster R-CNN 
  - [x] YOLO: you only look once
- [ ] segmentation?
- [ ] SSD
- [ ] CLIP-ResNet (read somewhere kinda interesting, mostprobably best ResNet till date? not sure)
- [ ] train something on COCO dataset? A good task? 
- [ ] Visualizing CNN techniques
  - [ ] DeepDream?



## Image Generation
- [ ] Pixel RNN (maybe if interested)
- [ ] VAE
- [ ] GAN
- [ ] Stable Diffusion
- [ ] DALL-E
- [ ] Vision QA models



## Reinforcement learning
- [ ] RF -- its a framework to teach agents
- [ ] DQN
- [ ] Policy Gradient Methods
- [ ] DPO



## Normalization and Regularization
- [ ] Quantization? Factorization?
- [ ] Weight Standardization
- [ ] Label Smoothing
- [ ] Filter Response Normalization 
- [ ] Normalization layers
  - [ ] BatchNorm
  - [ ] LayerNorm
  - [ ] GroupNorm
  - [ ] InstanceNorm
  - [ ] PowerNorm and weightNorm (are they good?)
  - [ ] RMSNorm--  most used, I think nowadays, used in LLaMA, Mistral, (Grok?), mostprobalbly also in GPT-4 (who knows)



## Some research papers
- [x] Understanding deeplearning requires rethinking generalization (research paper)-- See mnist_generalization [notebook](./mnist_generalization.ipynb)
- [ ] LM are Few Shot learners (GPT-3 paper: https://arxiv.org/pdf/2005.14165)
- [ ] The recurrent temporal restricted boltzmann machine (research paper)-- energy-model sounds interesting!
- [ ] Faster Training: Super-Convergence (research paper)
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
- [ ] https://arxiv.org/pdf/2401.00368
- [ ] https://arxiv.org/pdf/2401.01335
- [ ] https://arxiv.org/pdf/2404.12253



## Flops (Floating-Point Operations Per Second)
- [x] gemm in Python
- [ ] gemm in JAX
- [ ] gemm in c++


## Open Architectures
- [ ] Mixtral
- [ ] is Phi architecture open source?
- [ ] Grok-1
- [ ] OpenHermes by NousResearch


## Some resourceful repos
- [ ] https://github.com/coqui-ai/TTS?tab=readme-ov-file#model-implementations
