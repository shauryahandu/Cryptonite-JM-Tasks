import torch
import torch.nn.functional as F

from gpt import MiniGPT


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        num_layers=checkpoint["num_layers"],
        num_heads=checkpoint["num_heads"],
        context_length=checkpoint["context_length"],
        feedforward_dim=checkpoint["feedforward_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    string_to_index = checkpoint["string_to_index"]
    index_to_string = checkpoint["index_to_string"]

    return model, string_to_index, index_to_string, checkpoint["context_length"]


def encode_prompt(prompt, string_to_index):
    unknown_index = 0
    return [string_to_index.get(character, unknown_index) for character in prompt]


def decode_indices(indices, index_to_string):
    return "".join(index_to_string.get(index, "") for index in indices)


def greedy_decode(model, input_ids, context_length, max_new_tokens, device):
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        context_window = generated[-context_length:]
        input_tensor = torch.tensor([context_window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _, _ = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        generated.append(next_token)
    return generated


def temperature_sample_decode(model, input_ids, context_length, max_new_tokens, temperature, device):
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        context_window = generated[-context_length:]
        input_tensor = torch.tensor([context_window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _, _ = model(input_tensor)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-8)
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        generated.append(next_token)
    return generated


def top_k_sample_decode(model, input_ids, context_length, max_new_tokens, temperature, top_k, device):
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        context_window = generated[-context_length:]
        input_tensor = torch.tensor([context_window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _, _ = model(input_tensor)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-8)

        top_k_value = min(top_k, next_token_logits.size(-1))
        top_values, top_indices = torch.topk(next_token_logits, top_k_value)
        filtered_logits = torch.full_like(next_token_logits, float("-inf"))
        filtered_logits.scatter_(0, top_indices, top_values)

        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        generated.append(next_token)
    return generated


def top_p_sample_decode(model, input_ids, context_length, max_new_tokens, temperature, top_p, device):
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        context_window = generated[-context_length:]
        input_tensor = torch.tensor([context_window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _, _ = model(input_tensor)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-8)

        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        sorted_probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        cutoff_mask = cumulative_probabilities > top_p
        cutoff_mask[0] = False
        sorted_logits[cutoff_mask] = float("-inf")

        filtered_logits = torch.full_like(next_token_logits, float("-inf"))
        filtered_logits.scatter_(0, sorted_indices, sorted_logits)

        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        generated.append(next_token)
    return generated


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "best_model.pt"

    model, string_to_index, index_to_string, context_length = load_model_from_checkpoint(checkpoint_path, device)

    print("Loaded Model From Checkpoint")
    print(f"Context Length: {context_length}")
    print(f"Vocabulary Size: {len(string_to_index)}")

    prompts = [
        "The future of AI is",
        "Michael:",
        "Dwight:",
    ]

    max_new_tokens = 80

    for prompt in prompts:
        encoded_prompt = encode_prompt(prompt, string_to_index)

        greedy_result = greedy_decode(model, encoded_prompt, context_length, max_new_tokens, device)
        temperature_result = temperature_sample_decode(model, encoded_prompt, context_length, max_new_tokens, temperature=0.8, device=device)
        top_k_result = top_k_sample_decode(model, encoded_prompt, context_length, max_new_tokens, temperature=0.8, top_k=10, device=device)
        top_p_result = top_p_sample_decode(model, encoded_prompt, context_length, max_new_tokens, temperature=0.8, top_p=0.9, device=device)

        print(f"\nPrompt: {prompt}")
        print(f"Greedy Decoding: {decode_indices(greedy_result, index_to_string)}")
        print(f"Temperature Sampling: {decode_indices(temperature_result, index_to_string)}")
        print(f"Top K Sampling: {decode_indices(top_k_result, index_to_string)}")
        print(f"Top P Sampling: {decode_indices(top_p_result, index_to_string)}")
