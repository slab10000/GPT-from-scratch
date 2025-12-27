# GPT-from-scratch

A complete implementation of a GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. This project demonstrates how to build, train, and deploy a transformer-based language model similar to OpenAI's GPT architecture.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Architecture](#architecture)
- [Data Processing](#data-processing)
- [Model Components](#model-components)
- [Training](#training)
- [Inference](#inference)
- [Hyperparameters](#hyperparameters)
- [Usage Examples](#usage-examples)

## Overview

This project implements a GPT-style transformer model from the ground up, including:
- Multi-head self-attention mechanism
- Positional encoding
- Feed-forward networks
- Layer normalization
- Residual connections
- Dropout regularization

The model is trained on RFC (Request for Comments) documents and can generate text in a similar style.

## Features

- **Full GPT Architecture**: Implements all core components of the GPT model
- **Flexible Tokenization**: Supports both character-level and subword tokenization (tiktoken)
- **Training Infrastructure**: Includes checkpointing, mixed precision training, and learning rate scheduling
- **Efficient Training**: Optimized for GPU training with gradient accumulation
- **Text Generation**: Includes inference code for generating text from trained models

## Project Structure

```
GPT-from-scratch/
├── GPT_from_scratch.ipynb    # Main notebook with all code
└── README.md                  # This file
```

## Installation

### Requirements

```bash
pip install torch numpy tiktoken
```

### Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **tiktoken**: Fast BPE tokenizer (optional, can use character-level tokenization)

## Architecture

The model follows the standard GPT architecture:

```
Input Tokens
    ↓
Token Embeddings + Positional Embeddings
    ↓
[Transformer Block] × N layers
    ├── Multi-Head Self-Attention
    ├── Feed-Forward Network
    └── Layer Normalization + Residual Connections
    ↓
Final Layer Normalization
    ↓
Language Model Head (Linear Layer)
    ↓
Output Logits (Vocabulary Size)
```

## Data Processing

### Data Loading

The model loads text data from a file (default: RFC documents). The data is then tokenized and split into training and validation sets.

```python
# Load text data
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()
```

### Tokenization

The project supports two tokenization methods:

1. **Character-level Tokenization**: Maps each character to a unique integer
   ```python
   the_chars = sorted(list(set(text)))
   stoi = {ch: i for i, ch in enumerate(the_chars)}
   encode = lambda s: [stoi[c] for c in s]
   ```

2. **Subword Tokenization (tiktoken)**: Uses GPT-2's BPE tokenizer
   ```python
   tokenizer = tiktoken.get_encoding("gpt2")
   encode = lambda s: tokenizer.encode(s)
   ```

### Data Splitting

The dataset is split into 90% training and 10% validation:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

### Batch Generation

The `get_batch()` function creates random batches of sequences for training:

- Randomly selects starting positions in the dataset
- Creates input sequences of length `block_size`
- Creates target sequences (shifted by one position)
- Returns batches of shape `(batch_size, block_size)`

## Model Components

### 1. Attention Head (`Head`)

Implements a single attention head with scaled dot-product attention:

**Key Components:**
- **Query, Key, Value projections**: Linear transformations of input embeddings
- **Causal masking**: Prevents attention to future tokens (lower triangular matrix)
- **Scaled dot-product attention**: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
- **Dropout**: Regularization during training

**Forward Pass:**
1. Compute Q, K, V from input embeddings
2. Calculate attention weights: `wei = Q @ K^T / √head_size`
3. Apply causal mask (set future positions to -∞)
4. Apply softmax to get attention probabilities
5. Weighted aggregation: `out = wei @ V`

### 2. Multi-Head Attention (`MultiHeadAttention`)

Combines multiple attention heads in parallel:

- Creates `num_heads` attention heads
- Concatenates outputs from all heads
- Projects concatenated output back to embedding dimension
- Applies dropout for regularization

**Purpose**: Allows the model to attend to different types of information simultaneously.

### 3. Feed-Forward Network (`FeedForward`)

Two-layer MLP with ReLU activation:

```
Input (n_embd) → Linear(4 × n_embd) → ReLU → Linear(n_embd) → Output
```

**Purpose**: Provides non-linearity and allows the model to process information from attention layers.

### 4. Transformer Block (`Block`)

A complete transformer block combining attention and feed-forward layers:

**Structure:**
```
x → LayerNorm → MultiHeadAttention → + (residual) → LayerNorm → FeedForward → + (residual) → output
```

**Key Features:**
- **Residual connections**: Helps with gradient flow and training stability
- **Layer normalization**: Applied before each sub-layer (pre-norm architecture)
- **Dropout**: Applied in attention and feed-forward layers

### 5. GPT Model (`GPTModel`)

The complete GPT model architecture:

**Components:**

1. **Token Embedding Table**: Maps token IDs to dense vectors
   ```python
   self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
   ```

2. **Positional Embedding Table**: Adds positional information
   ```python
   self.pos_emb_table = nn.Embedding(block_size, n_embd)
   ```

3. **Transformer Blocks**: Stack of N transformer blocks
   ```python
   self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
   ```

4. **Final Layer Normalization**: Applied before the language model head

5. **Language Model Head**: Projects embeddings to vocabulary logits
   ```python
   self.lm_head = nn.Linear(n_embd, vocab_size)
   ```

**Forward Pass:**
1. Embed tokens and add positional encodings
2. Pass through transformer blocks
3. Apply final layer normalization
4. Project to vocabulary logits
5. Compute cross-entropy loss (if targets provided)

**Generation Method:**
- Autoregressively generates tokens one at a time
- Uses the last `block_size` tokens as context
- Samples next token from probability distribution
- Appends to sequence and repeats

## Training

### Training Loop

The training process includes:

1. **Gradient Accumulation**: Accumulates gradients over multiple batches for effective larger batch sizes
2. **Mixed Precision Training**: Uses automatic mixed precision (AMP) for faster training on GPUs
3. **Learning Rate Scheduling**: Cosine annealing with warmup
4. **Checkpointing**: Saves model checkpoints periodically
5. **Validation**: Evaluates on validation set at regular intervals

### Learning Rate Schedule

The learning rate follows a warmup + cosine decay schedule:

- **Warmup phase**: Linear increase from 0 to `learning_rate` over `warmup_steps`
- **Decay phase**: Cosine decay from `learning_rate` to 10% of `learning_rate`

### Checkpointing

The training script saves:
- **Latest checkpoint**: Most recent training state
- **Best checkpoint**: Model with lowest validation loss
- **Periodic checkpoints**: Saved at regular intervals

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Gradient scaler state (for mixed precision)
- Training step number
- Best validation loss
- Model configuration

### Loss Estimation

The `estimate_loss()` function:
- Evaluates the model on multiple batches
- Computes average loss on training and validation sets
- Uses `@torch.no_grad()` for efficiency
- Sets model to eval mode during evaluation

## Inference

### Loading a Trained Model

1. Load checkpoint file
2. Reconstruct model architecture from saved configuration
3. Load model weights
4. Set model to evaluation mode

### Text Generation

The generation process:

1. Encode input prompt to token IDs
2. Convert to tensor and add batch dimension
3. Call `model.generate()` with:
   - Input context
   - Maximum number of tokens to generate
4. Decode generated token IDs back to text

**Generation Strategy:**
- Uses the last `block_size` tokens as context (crops if longer)
- Samples from probability distribution (not greedy)
- Autoregressive: each new token depends on all previous tokens

## Hyperparameters

### Model Architecture

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `n_embd` | 512 | Embedding dimension |
| `n_head` | 8 | Number of attention heads |
| `n_layer` | 6 | Number of transformer blocks |
| `dropout` | 0.2 | Dropout probability |
| `vocab_size` | 50257 (tiktoken) or variable (char-level) | Vocabulary size |

### Training

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `block_size` | 512 | Context window size (sequence length) |
| `batch_size` | 64 | Batch size |
| `max_iters` | 135,000 | Maximum training iterations |
| `learning_rate` | 3e-4 | Initial learning rate |
| `eval_interval` | 500 | Evaluation frequency |
| `eval_iters` | 200 | Number of batches for evaluation |
| `grad_accum` | 2 | Gradient accumulation steps |

### Learning Rate Schedule

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `warmup_steps` | 2,000 | Warmup iterations |
| `final_lr_ratio` | 0.1 | Final LR as fraction of initial LR |

## Usage Examples

### Training a Model

```python
# Set hyperparameters
block_size = 512
batch_size = 64
n_embd = 512
n_head = 8
n_layer = 6

# Load and prepare data
with open('data.txt', 'r') as f:
    text = f.read()

# Tokenize and create datasets
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]

# Initialize model
model = GPTModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Generating Text

```python
# Load trained model
ckpt = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()

# Generate text
context = encode("What is IP?")
context_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(context_tensor, max_new_tokens=400)
output_text = decode(generated[0].tolist())
print(output_text)
```

## Key Concepts Explained

### Self-Attention

Self-attention allows each position in the sequence to attend to all previous positions. The attention mechanism computes:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

The attention score between positions i and j is: `Q_i · K_j`, which determines how much position i should attend to position j.

### Causal Masking

Causal masking ensures the model can only attend to previous tokens (not future ones), making it suitable for autoregressive generation. This is implemented using a lower triangular matrix.

### Positional Encoding

Since transformers have no inherent notion of sequence order, positional encodings are added to token embeddings to provide information about token positions in the sequence.

### Residual Connections

Residual connections (skip connections) allow gradients to flow directly through the network, helping with training deep models. They enable the model to learn identity mappings when needed.

### Layer Normalization

Layer normalization stabilizes training by normalizing activations across the embedding dimension. It's applied before each sub-layer in the transformer block.

## Performance Considerations

- **GPU Training**: The code automatically detects and uses CUDA if available
- **Mixed Precision**: Uses automatic mixed precision (AMP) to reduce memory usage and speed up training
- **Gradient Accumulation**: Allows effective larger batch sizes without increasing memory requirements
- **Efficient Batching**: Random batch sampling for better generalization

## Future Improvements

Potential enhancements:
- [ ] Add support for different model sizes (GPT-2 small, medium, large)
- [ ] Implement gradient checkpointing for memory efficiency
- [ ] Add support for distributed training
- [ ] Implement different sampling strategies (top-k, top-p, temperature)
- [ ] Add support for fine-tuning on specific tasks
- [ ] Implement model quantization for inference

## License

This project is for educational purposes. Feel free to use and modify as needed.

## Acknowledgments

This implementation is inspired by:
- The original GPT paper: "Improving Language Understanding by Generative Pre-Training"
- Andrej Karpathy's "Let's build GPT" series
- The transformer architecture from "Attention is All You Need"

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
