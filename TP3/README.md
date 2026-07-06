# Decoder Only Transformers: Building GPT From Scratch

## Project Overview

This project implements a decoder only transformer language model, MiniGPT, entirely from first principles using PyTorch primitives. It progresses from classical n-gram language models through byte pair encoding tokenization, positional encodings, scaled dot product attention, multi head attention, transformer decoder blocks, a complete GPT architecture, a training pipeline, text generation with multiple decoding strategies, and attention visualization.

## Project Structure

| File | Description |
|---|---|
| `ngram_lm.py` | Unigram, bigram, and trigram language models with Laplace smoothing and perplexity evaluation |
| `bpe_tokenizer.py` | Standalone byte pair encoding tokenizer with comparisons to word and character level tokenization |
| `positional_encoding.py` | Sinusoidal and learned positional encodings with visualizations |
| `attention.py` | Scaled dot product attention implemented from scratch with mask and dropout support |
| `masking.py` | Causal mask and triangular attention mask construction and visualization |
| `multihead.py` | Multi head attention with projection layers, head splitting, and head combining |
| `decoder_block.py` | Full transformer decoder block with residual connections, layer normalization, and a feed forward network |
| `gpt.py` | MiniGPT architecture combining token embeddings, positional embeddings, a decoder stack, and a language modeling head |
| `train.py` | Training pipeline with dataset loading, vocabulary construction, AdamW optimization, gradient clipping, validation, and checkpointing |
| `generate.py` | Greedy, temperature, top k, and top p decoding for text generation from a trained checkpoint |
| `attention_visualization.py` | Extraction and visualization of attention matrices from a trained model |

## Running The Project

Each file is independently executable and contains a demonstration section under `if __name__ == "__main__":` that prints outputs proving the implementation works.

```
python3 ngram_lm.py
python3 bpe_tokenizer.py
python3 positional_encoding.py
python3 attention.py
python3 masking.py
python3 multihead.py
python3 decoder_block.py
python3 gpt.py
python3 train.py
python3 generate.py
python3 attention_visualization.py
```

`train.py` must be run before `generate.py` and `attention_visualization.py`, since both depend on the `best_model.pt` checkpoint produced during training.

On Kaggle, `ngram_lm.py` and `train.py` will automatically use the Tiny Shakespeare character modeling dataset at the configured Kaggle paths. When those paths are not available, both files fall back to an embedded public domain Shakespeare excerpt so that every file remains fully executable in any environment.

## Conceptual Questions

### Bigram Language Model vs GPT

A bigram model estimates the probability of the next token using only the single preceding token, with probabilities derived from raw frequency counts and a smoothing scheme such as Laplace smoothing. It has no notion of representation learning, no parameters beyond a count table, and its effective context window is fixed at one token, which makes it unable to capture long range dependencies or semantic relationships between distant words.

GPT is an autoregressive neural network that also predicts the next token given previous tokens, but it represents each token as a dense vector, propagates information through many layers of self attention and feed forward transformations, and conditions its prediction on the entire preceding context up to its maximum context length rather than a single token. Self attention allows GPT to learn which earlier tokens are relevant to the current prediction, and its parameters are learned through gradient based optimization on large corpora rather than fixed frequency counts. As a result GPT can model long range syntactic and semantic structure, generalize to unseen sequences through learned embeddings, and produce far lower perplexity than a bigram model on natural language, at the cost of dramatically higher computational and data requirements.

### Hidden Markov Model vs GPT

A Hidden Markov Model assumes a sequence of discrete hidden states that evolve according to a Markov process, where each observed token is emitted from a hidden state according to an emission distribution, and the hidden state depends only on a fixed number of previous states under the Markov assumption. Inference and training rely on algorithms such as the forward backward algorithm and the Viterbi algorithm, and the model is fundamentally limited by the Markov assumption and the size of its discrete state space.

GPT does not maintain an explicit discrete hidden state sequence. Instead it maintains continuous hidden representations at every layer and position that are computed through self attention over the entire visible context, allowing it to condition on arbitrarily distant tokens within its context window rather than a fixed order Markov window. GPT is trained end to end with backpropagation on a next token prediction objective rather than an expectation maximization style procedure, and its representational capacity scales with the number of parameters and training data rather than the number of discrete hidden states. In practice this allows GPT to capture far richer contextual and semantic dependencies than an HMM.

### BERT vs GPT

BERT is an encoder only transformer trained with a masked language modeling objective, where a subset of input tokens are replaced with a mask token and the model must predict the original tokens using bidirectional context from both left and right of each masked position. This bidirectional attention makes BERT well suited to producing rich contextual representations for classification, extraction, and other understanding tasks, but it is not naturally suited to autoregressive generation because every position can attend to every other position without any causal restriction.

GPT is a decoder only transformer trained with a causal, left to right next token prediction objective, and it uses a causal attention mask so that each position can only attend to itself and earlier positions. This causal structure is what allows GPT to generate text one token at a time in an autoregressive manner, since at generation time only past tokens are available. The core architectural difference is therefore the attention masking strategy and training objective, bidirectional and masked for BERT versus causal and autoregressive for GPT, which in turn determines that BERT excels at representation learning for understanding tasks while GPT excels at open ended text generation.

### Viterbi Decoding vs Autoregressive Decoding

Viterbi decoding is a dynamic programming algorithm used with models such as Hidden Markov Models to find the single most probable sequence of hidden states given an observed sequence. It works by exploring the full state space at every time step, keeping track of the best scoring path into every state, and backtracking once the entire sequence has been processed to recover the globally optimal state sequence. This produces an exact, globally optimal solution under the Markov assumption for the objective it is optimizing, but its computational cost scales with the number of possible states and requires the full observation sequence before decoding can be finalized.

Autoregressive decoding, as used in GPT, generates tokens one at a time, feeding each generated token back into the model as part of the context for generating the next token. There is no explicit dynamic programming search over a state space and no guarantee of global optimality, since strategies such as greedy decoding, temperature sampling, top k sampling, and top p sampling all make local decisions at each step conditioned on the tokens generated so far. This local, sequential decision process is far more computationally efficient for very large vocabularies and long sequences than an exhaustive Viterbi style search, and it also naturally supports open ended generation since it does not require the sequence length to be known in advance, but it forgoes the global optimality guarantee that Viterbi decoding provides for its narrower class of models.

## Reflection

Building each component of a decoder only transformer independently, from n-gram language models through byte pair encoding, positional encodings, scaled dot product attention, multi head attention, and a full decoder stack, clarifies how each architectural choice addresses a specific limitation of the previous approach. N-gram models expose the sparsity and limited context problems that motivate learned dense representations. Byte pair encoding demonstrates how a fixed size subword vocabulary can balance the tradeoffs between word level and character level tokenization, keeping sequence lengths manageable while handling out of vocabulary words gracefully. Positional encodings reveal why transformers, which otherwise have no inherent notion of order, need an explicit mechanism to inject sequence position information. Scaled dot product attention and its extension to multiple heads show how a model can learn to weigh the relevance of every previous token when forming a representation for the current position, and stacking this inside residual, normalized decoder blocks demonstrates how depth and gradient flow are managed in practice. Finally, assembling MiniGPT and training it end to end on real text makes concrete how all of these pieces combine into a single differentiable system that can be optimized purely from a next token prediction objective, and comparing the resulting causal, autoregressive generation process to classical decoding algorithms such as Viterbi highlights the tradeoff between exact combinatorial optimality and the scalable, flexible generation that modern language models rely on.
