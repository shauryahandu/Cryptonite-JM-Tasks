import torch
import torch.nn as nn

from multihead import MultiHeadAttention


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)

        self.feed_forward = FeedForwardNetwork(embedding_dim=embedding_dim, hidden_dim=feedforward_dim, dropout=dropout)
        self.feed_forward_layer_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output, attention_weights = self.attention(x, x, x, mask=mask)
        x = x + self.attention_dropout(attention_output)
        x = self.attention_layer_norm(x)

        feed_forward_output = self.feed_forward(x)
        x = x + self.feed_forward_dropout(feed_forward_output)
        x = self.feed_forward_layer_norm(x)

        return x, attention_weights


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 12
    embedding_dim = 64
    num_heads = 4
    feedforward_dim = 256

    decoder_block = DecoderBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=0.1,
    )

    x = torch.randn(batch_size, sequence_length, embedding_dim)
    causal_mask = torch.tril(torch.ones(sequence_length, sequence_length)).unsqueeze(0)

    output, attention_weights = decoder_block(x, mask=causal_mask)

    print("Decoder Block Demonstration")
    print(f"Input Shape: {tuple(x.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")
    print(f"Attention Weights Shape: {tuple(attention_weights.shape)}")

    total_parameters = sum(parameter.numel() for parameter in decoder_block.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in decoder_block.parameters() if parameter.requires_grad)

    print(f"\nTotal Parameters: {total_parameters}")
    print(f"Trainable Parameters: {trainable_parameters}")

    decoder_block.eval()
    with torch.no_grad():
        eval_output, eval_attention_weights = decoder_block(x, mask=causal_mask)

    print(f"\nEvaluation Mode Output Shape: {tuple(eval_output.shape)}")
    print(f"Output Mean: {output.mean().item():.6f}")
    print(f"Output Std: {output.std().item():.6f}")
