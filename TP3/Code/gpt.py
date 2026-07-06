import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder_block import DecoderBlock


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        context_length=128,
        feedforward_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.language_model_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.register_buffer("causal_mask_template", torch.tril(torch.ones(context_length, context_length)))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.language_model_head.weight, mean=0.0, std=0.02)

    def get_causal_mask(self, sequence_length, device):
        mask = self.causal_mask_template[:sequence_length, :sequence_length].to(device)
        return mask.unsqueeze(0)

    def forward(self, input_ids, targets=None):
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.context_length:
            raise ValueError("sequence_length exceeds context_length")

        device = input_ids.device
        positions = torch.arange(0, sequence_length, device=device).unsqueeze(0)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)

        x = self.embedding_dropout(token_embeddings + position_embeddings)

        causal_mask = self.get_causal_mask(sequence_length, device)

        attention_weights_per_layer = []
        for decoder_block in self.decoder_blocks:
            x, attention_weights = decoder_block(x, mask=causal_mask)
            attention_weights_per_layer.append(attention_weights)

        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, attention_weights_per_layer

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 65
    batch_size = 4
    sequence_length = 32

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        context_length=128,
        feedforward_dim=512,
        dropout=0.1,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))

    logits, loss, attention_weights_per_layer = model(input_ids, targets)

    print("MiniGPT Demonstration")
    print(f"Input Shape: {tuple(input_ids.shape)}")
    print(f"Logits Shape: {tuple(logits.shape)}")
    print(f"Loss Value: {loss.item():.6f}")
    print(f"Number Of Decoder Layers: {model.num_layers}")
    print(f"Number Of Attention Heads: {model.num_heads}")
    print(f"Total Trainable Parameters: {model.count_parameters()}")

    print("\nAttention Weights Per Layer")
    for layer_index, attention_weights in enumerate(attention_weights_per_layer):
        print(f"Layer {layer_index + 1} Attention Weights Shape: {tuple(attention_weights.shape)}")

    model.eval()
    with torch.no_grad():
        eval_logits, eval_loss, _ = model(input_ids, targets)

    print(f"\nEvaluation Mode Logits Shape: {tuple(eval_logits.shape)}")
    print(f"Evaluation Mode Loss Value: {eval_loss.item():.6f}")

    inference_logits, inference_loss, _ = model(input_ids)
    print(f"\nInference Without Targets Logits Shape: {tuple(inference_logits.shape)}")
    print(f"Inference Without Targets Loss Is None: {inference_loss is None}")
