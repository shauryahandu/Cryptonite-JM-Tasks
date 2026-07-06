import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_sequence_length=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        position = torch.arange(0, max_sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))

        encoding = torch.zeros(max_sequence_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        sequence_length = x.size(1)
        return x + self.encoding[:, :sequence_length, :]

    def get_encoding_matrix(self, sequence_length):
        return self.encoding[0, :sequence_length, :]


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_sequence_length=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(max_sequence_length, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        sequence_length = x.size(1)
        positions = torch.arange(0, sequence_length, device=x.device).unsqueeze(0)
        return x + self.embedding(positions)

    def get_encoding_matrix(self, sequence_length):
        positions = torch.arange(0, sequence_length)
        return self.embedding(positions).detach()


def visualize_positional_encoding(encoding_matrix, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(encoding_matrix.numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Value")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    embedding_dim = 64
    sequence_length = 50
    batch_size = 2

    dummy_input = torch.zeros(batch_size, sequence_length, embedding_dim)

    sinusoidal_module = SinusoidalPositionalEncoding(embedding_dim=embedding_dim, max_sequence_length=512)
    learned_module = LearnedPositionalEmbedding(embedding_dim=embedding_dim, max_sequence_length=512)

    sinusoidal_output = sinusoidal_module(dummy_input)
    learned_output = learned_module(dummy_input)

    print("Sinusoidal Positional Encoding")
    print(f"Input Shape: {tuple(dummy_input.shape)}")
    print(f"Output Shape: {tuple(sinusoidal_output.shape)}")
    print(f"First Positional Vector: {sinusoidal_module.get_encoding_matrix(sequence_length)[0][:8]}")

    print("\nLearned Positional Embedding")
    print(f"Input Shape: {tuple(dummy_input.shape)}")
    print(f"Output Shape: {tuple(learned_output.shape)}")
    print(f"First Positional Vector: {learned_module.get_encoding_matrix(sequence_length)[0][:8]}")

    sinusoidal_matrix = sinusoidal_module.get_encoding_matrix(sequence_length)
    learned_matrix = learned_module.get_encoding_matrix(sequence_length)

    visualize_positional_encoding(sinusoidal_matrix, "Sinusoidal Positional Encoding", "sinusoidal_encoding.png")
    visualize_positional_encoding(learned_matrix, "Learned Positional Embedding (Initial)", "learned_encoding.png")

    print("\nSaved sinusoidal_encoding.png")
    print("Saved learned_encoding.png")
