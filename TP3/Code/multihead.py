import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        embedding_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(embedding_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.query_projection, self.key_projection, self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def split_heads(self, x):
        batch_size, sequence_length, embedding_dim = x.shape
        x = x.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, num_heads, sequence_length, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, sequence_length, num_heads * head_dim)

    def forward(self, query, key, value, mask=None):
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)

        query_heads = self.split_heads(query_projected)
        key_heads = self.split_heads(key_projected)
        value_heads = self.split_heads(value_projected)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        attention_output, attention_weights = self.attention(query_heads, key_heads, value_heads, mask=mask)

        combined_output = self.combine_heads(attention_output)
        output = self.output_projection(combined_output)
        output = self.dropout(output)

        return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 10
    embedding_dim = 32
    num_heads = 4

    multihead_attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, dropout=0.1)

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    causal_mask = torch.tril(torch.ones(sequence_length, sequence_length)).unsqueeze(0)

    output, attention_weights = multihead_attention(x, x, x, mask=causal_mask)

    print("Multi Head Attention Demonstration")
    print(f"Input Shape: {tuple(x.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")
    print(f"Attention Weights Shape: {tuple(attention_weights.shape)}")
    print(f"Number Of Heads: {multihead_attention.num_heads}")
    print(f"Head Dimension: {multihead_attention.head_dim}")

    split_example = multihead_attention.split_heads(x)
    combined_example = multihead_attention.combine_heads(split_example)

    print(f"\nSplit Heads Shape: {tuple(split_example.shape)}")
    print(f"Combined Heads Shape: {tuple(combined_example.shape)}")
    print(f"Split Then Combine Matches Original: {torch.allclose(combined_example, x)}")

    first_row_nonzero = int((attention_weights[0, 0, 0] > 0).sum().item())
    last_row_nonzero = int((attention_weights[0, 0, -1] > 0).sum().item())

    print(f"\nCausal Mask Check First Row Non Zero Entries: {first_row_nonzero}")
    print(f"Causal Mask Check Last Row Non Zero Entries: {last_row_nonzero}")

    multihead_attention.eval()
    with torch.no_grad():
        eval_output, _ = multihead_attention(x, x, x, mask=causal_mask)
    print(f"\nEvaluation Mode Output Shape: {tuple(eval_output.shape)}")
