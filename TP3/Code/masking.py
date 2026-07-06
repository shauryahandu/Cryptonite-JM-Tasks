import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_causal_mask(sequence_length):
    mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.float32))
    return mask


def generate_triangular_attention_mask(sequence_length):
    mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.float32), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def apply_causal_mask_to_scores(scores, mask):
    return scores.masked_fill(mask == 0, float("-inf"))


def visualize_mask(mask, output_path, title="Causal Mask"):
    matrix = mask.detach().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap="gray_r")
    plt.colorbar(label="Mask Value")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    sequence_length = 12

    causal_mask = generate_causal_mask(sequence_length)
    triangular_mask = generate_triangular_attention_mask(sequence_length)

    print("GPT Causal Mask")
    print(f"Shape: {tuple(causal_mask.shape)}")
    print(causal_mask)

    print("\nTriangular Attention Mask")
    print(f"Shape: {tuple(triangular_mask.shape)}")
    print(triangular_mask)

    dummy_scores = torch.randn(sequence_length, sequence_length)
    masked_scores = apply_causal_mask_to_scores(dummy_scores, causal_mask)

    print("\nDummy Attention Scores After Applying Causal Mask")
    print(masked_scores)

    visualize_mask(causal_mask, "causal_mask.png", title="GPT Causal Mask")
    print("\nSaved causal_mask.png")
