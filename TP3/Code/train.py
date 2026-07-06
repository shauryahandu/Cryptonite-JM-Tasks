import os
import csv
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gpt import MiniGPT

csv.field_size_limit(10000000)


class CharDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, index):
        input_sequence = self.data[index:index + self.context_length]
        target_sequence = self.data[index + 1:index + self.context_length + 1]
        return input_sequence, target_sequence


def read_csv_text_column(path):
    texts = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                if "text" in row and row["text"]:
                    texts.append(row["text"])
    return "\n".join(texts)


def load_text_from_kaggle(directory_path):
    train_csv_path = os.path.join(directory_path, "train.csv")
    validation_csv_path = os.path.join(directory_path, "validation.csv")
    train_text = read_csv_text_column(train_csv_path)
    validation_text = read_csv_text_column(validation_csv_path)
    return train_text, validation_text


def load_text_from_first_available_directory(candidate_directories):
    for directory_path in candidate_directories:
        if directory_path is None:
            continue
        train_text, validation_text = load_text_from_kaggle(directory_path)
        if train_text and validation_text:
            print(f"Loaded Tiny Shakespeare Dataset From: {directory_path}")
            return train_text, validation_text
    return "", ""


def build_vocabulary(text):
    unique_characters = sorted(list(set(text)))
    string_to_index = {character: index for index, character in enumerate(unique_characters)}
    index_to_string = {index: character for character, index in string_to_index.items()}
    return string_to_index, index_to_string


def encode_text(text, string_to_index):
    return torch.tensor([string_to_index[character] for character in text], dtype=torch.long)


def evaluate_model(model, data_loader, device, max_steps=None):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            if max_steps is not None and total_batches >= max_steps:
                break
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            _, loss, _ = model(input_batch, target_batch)
            total_loss += loss.item()
            total_batches += 1
    if total_batches == 0:
        return float("inf")
    return total_loss / total_batches


FALLBACK_TRAIN_TEXT = (
    "Shall I compare thee to a summer's day\n"
    "Thou art more lovely and more temperate\n"
    "Rough winds do shake the darling buds of May\n"
    "And summer's lease hath all too short a date\n"
    "Sometime too hot the eye of heaven shines\n"
    "And often is his gold complexion dimmed\n"
    "And every fair from fair sometime declines\n"
    "By chance or nature's changing course untrimmed\n"
    "But thy eternal summer shall not fade\n"
    "Nor lose possession of that fair thou owest\n"
    "Nor shall death brag thou wanderest in his shade\n"
    "When in eternal lines to time thou growest\n"
    "So long as men can breathe or eyes can see\n"
    "So long lives this and this gives life to thee\n"
    "To be or not to be that is the question\n"
    "Whether tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them to die to sleep\n"
    "No more and by a sleep to say we end\n"
    "The heartache and the thousand natural shocks\n"
    "That flesh is heir to tis a consummation\n"
    "Devoutly to be wished to die to sleep\n"
    "To sleep perchance to dream ay there is the rub\n"
    "For in that sleep of death what dreams may come\n"
    "When we have shuffled off this mortal coil\n"
    "Must give us pause there is the respect\n"
    "That makes calamity of so long life\n"
) * 4

FALLBACK_VALIDATION_TEXT = (
    "For who would bear the whips and scorns of time\n"
    "The oppressor's wrong the proud man's contumely\n"
    "The pangs of despised love the law's delay\n"
    "The insolence of office and the spurns\n"
    "That patient merit of the unworthy takes\n"
    "When he himself might his quietus make\n"
    "With a bare bodkin who would fardels bear\n"
) * 4


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kaggle_directory = "/kaggle/input/datasets/thedevastator/the-bards-best-a-character-modeling-dataset/"
    environment_directory = os.environ.get("DATASET_DIR")
    local_directory = os.path.join(os.getcwd(), "data")

    candidate_directories = [kaggle_directory, environment_directory, local_directory]
    train_text, validation_text = load_text_from_first_available_directory(candidate_directories)

    if not train_text or not validation_text:
        print("Tiny Shakespeare Dataset Not Found In Any Candidate Directory")
        print("Set The DATASET_DIR Environment Variable To A Folder Containing train.csv And validation.csv")
        print("Falling Back To Embedded Sample Text")

    if not train_text:
        train_text = FALLBACK_TRAIN_TEXT
    if not validation_text:
        validation_text = FALLBACK_VALIDATION_TEXT

    combined_text = train_text + validation_text
    string_to_index, index_to_string = build_vocabulary(combined_text)
    vocab_size = len(string_to_index)

    train_data = encode_text(train_text, string_to_index)
    validation_data = encode_text(validation_text, string_to_index)

    default_context_length = 128
    context_length = min(default_context_length, max(8, len(train_data) - 1))

    embedding_dim = 128
    num_layers = 4
    num_heads = 4
    feedforward_dim = 512
    dropout = 0.1
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 3
    max_train_steps_per_epoch = 40
    max_validation_steps_per_epoch = 20
    gradient_clip_value = 1.0

    train_dataset = CharDataset(train_data, context_length)
    validation_dataset = CharDataset(validation_data, context_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        context_length=context_length,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Training Configuration")
    print(f"Device: {device}")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Context Length: {context_length}")
    print(f"Training Sequences: {len(train_dataset)}")
    print(f"Validation Sequences: {len(validation_dataset)}")
    print(f"Total Model Parameters: {model.count_parameters()}")

    best_validation_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_batches = 0

        for input_batch, target_batch in train_loader:
            if total_train_batches >= max_train_steps_per_epoch:
                break

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            _, loss, _ = model(input_batch, target_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_batches += 1

        average_train_loss = total_train_loss / max(1, total_train_batches)
        average_validation_loss = evaluate_model(model, validation_loader, device, max_steps=max_validation_steps_per_epoch)

        train_perplexity = math.exp(average_train_loss) if average_train_loss < 20 else float("inf")
        validation_perplexity = math.exp(average_validation_loss) if average_validation_loss < 20 else float("inf")

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Training Loss: {average_train_loss:.4f}")
        print(f"Validation Loss: {average_validation_loss:.4f}")
        print(f"Training Perplexity: {train_perplexity:.4f}")
        print(f"Validation Perplexity: {validation_perplexity:.4f}")

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "string_to_index": string_to_index,
                "index_to_string": index_to_string,
                "embedding_dim": embedding_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "context_length": context_length,
                "feedforward_dim": feedforward_dim,
                "dropout": dropout,
                "validation_loss": average_validation_loss,
            }
            torch.save(checkpoint, "best_model.pt")
            print(f"Saved Checkpoint With Validation Loss: {average_validation_loss:.4f}")

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.4f} seconds")
