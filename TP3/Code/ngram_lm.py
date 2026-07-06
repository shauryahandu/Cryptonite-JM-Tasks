import os
import re
import csv
import math
import time
import collections

csv.field_size_limit(10000000)


class NGramLanguageModel:
    def __init__(self, n, smoothing_k=1.0):
        self.n = n
        self.smoothing_k = smoothing_k
        self.vocab = set()
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        self.vocab_size = 0

    def build_vocab(self, token_sequences):
        for tokens in token_sequences:
            self.vocab.update(tokens)
        self.vocab.add("<unk>")
        self.vocab.add("<s>")
        self.vocab.add("</s>")
        self.vocab_size = len(self.vocab)

    def _pad(self, tokens):
        return ["<s>"] * (self.n - 1) + list(tokens) + ["</s>"]

    def train(self, token_sequences):
        self.build_vocab(token_sequences)
        for tokens in token_sequences:
            padded = self._pad(tokens)
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1:i])
                word = padded[i]
                self.ngram_counts[context + (word,)] += 1
                self.context_counts[context] += 1

    def _normalize_word(self, word):
        if word in self.vocab:
            return word
        return "<unk>"

    def get_probability(self, context, word):
        context = tuple(self._normalize_word(w) for w in context[-(self.n - 1):]) if self.n > 1 else tuple()
        word = self._normalize_word(word)
        ngram = context + (word,)
        numerator = self.ngram_counts.get(ngram, 0) + self.smoothing_k
        denominator = self.context_counts.get(context, 0) + self.smoothing_k * self.vocab_size
        return numerator / denominator

    def sequence_log_probability(self, tokens):
        padded = self._pad(tokens)
        log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = padded[i - self.n + 1:i]
            word = padded[i]
            prob = self.get_probability(context, word)
            log_prob += math.log(prob)
        return log_prob

    def perplexity(self, token_sequences):
        total_log_prob = 0.0
        total_tokens = 0
        for tokens in token_sequences:
            total_log_prob += self.sequence_log_probability(tokens)
            total_tokens += len(tokens) + 1
        if total_tokens == 0:
            return float("inf")
        avg_log_prob = total_log_prob / total_tokens
        return math.exp(-avg_log_prob)


class UnigramLanguageModel(NGramLanguageModel):
    def __init__(self, smoothing_k=1.0):
        super().__init__(n=1, smoothing_k=smoothing_k)


class BigramLanguageModel(NGramLanguageModel):
    def __init__(self, smoothing_k=1.0):
        super().__init__(n=2, smoothing_k=smoothing_k)


class TrigramLanguageModel(NGramLanguageModel):
    def __init__(self, smoothing_k=1.0):
        super().__init__(n=3, smoothing_k=smoothing_k)


def read_csv_lines(path):
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                if "text" in row and row["text"]:
                    for line in row["text"].split("\n"):
                        stripped_line = line.strip()
                        if stripped_line:
                            lines.append(stripped_line)
    return lines


def load_dataset_from_first_available_path(candidate_paths, fallback_texts):
    for path in candidate_paths:
        if not path:
            continue
        lines = read_csv_lines(path)
        if lines:
            print(f"Loaded Dataset From: {path}")
            return lines
    return fallback_texts


def load_dataset(path, fallback_texts):
    return load_dataset_from_first_available_path([path], fallback_texts)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = text.split()
    return tokens


def build_token_sequences(texts):
    return [preprocess_text(t) for t in texts if t.strip()]


FALLBACK_TRAIN_TEXTS = [
    "Shall I compare thee to a summer's day",
    "Thou art more lovely and more temperate",
    "Rough winds do shake the darling buds of May",
    "And summer's lease hath all too short a date",
    "Sometime too hot the eye of heaven shines",
    "And often is his gold complexion dimmed",
    "And every fair from fair sometime declines",
    "By chance or nature's changing course untrimmed",
    "But thy eternal summer shall not fade",
    "Nor lose possession of that fair thou owest",
    "Nor shall death brag thou wanderest in his shade",
    "When in eternal lines to time thou growest",
    "So long as men can breathe or eyes can see",
    "So long lives this and this gives life to thee",
    "To be or not to be that is the question",
    "Whether tis nobler in the mind to suffer",
    "The slings and arrows of outrageous fortune",
    "Or to take arms against a sea of troubles",
    "And by opposing end them to die to sleep",
    "No more and by a sleep to say we end",
]

FALLBACK_VAL_TEXTS = [
    "The heartache and the thousand natural shocks",
    "That flesh is heir to tis a consummation",
    "Devoutly to be wished to die to sleep",
    "To sleep perchance to dream ay there's the rub",
    "For in that sleep of death what dreams may come",
    "When we have shuffled off this mortal coil",
    "Must give us pause there's the respect",
    "That makes calamity of so long life",
    "For who would bear the whips and scorns of time",
    "The oppressor's wrong the proud man's contumely",
]


if __name__ == "__main__":
    start_time = time.time()

    kaggle_train_path = "/kaggle/input/datasets/thedevastator/the-bards-best-a-character-modeling-dataset/train.csv"
    kaggle_validation_path = "/kaggle/input/datasets/thedevastator/the-bards-best-a-character-modeling-dataset/validation.csv"
    environment_directory = os.environ.get("DATASET_DIR")
    local_directory = os.path.join(os.getcwd(), "data")

    train_candidates = [
        kaggle_train_path,
        os.path.join(environment_directory, "train.csv") if environment_directory else None,
        os.path.join(local_directory, "train.csv"),
    ]
    validation_candidates = [
        kaggle_validation_path,
        os.path.join(environment_directory, "validation.csv") if environment_directory else None,
        os.path.join(local_directory, "validation.csv"),
    ]

    train_texts = load_dataset_from_first_available_path(train_candidates, FALLBACK_TRAIN_TEXTS)
    val_texts = load_dataset_from_first_available_path(validation_candidates, FALLBACK_VAL_TEXTS)

    train_sequences = build_token_sequences(train_texts)
    val_sequences = build_token_sequences(val_texts)

    print("Dataset Summary")
    print(f"Training Sequences: {len(train_sequences)}")
    print(f"Validation Sequences: {len(val_sequences)}")

    models = collections.OrderedDict()
    models["Unigram"] = UnigramLanguageModel()
    models["Bigram"] = BigramLanguageModel()
    models["Trigram"] = TrigramLanguageModel()

    for name, model in models.items():
        model.train(train_sequences)
        print(f"\n{name} Model Vocabulary Size: {model.vocab_size}")

    sample_context_word_pairs = [
        ([], "thou"),
        (["thou"], "art"),
        (["to", "be"], "or"),
        (["shall", "i", "compare"], "thee"),
    ]

    for name, model in models.items():
        print(f"\n{name} Model Example Probabilities")
        for context, word in sample_context_word_pairs:
            prob = model.get_probability(context, word)
            context_display = " ".join(context) if context else "<start>"
            print(f"P({word} | {context_display}) = {prob:.6f}")

    print("\nValidation Perplexities")
    for name, model in models.items():
        ppl = model.perplexity(val_sequences)
        print(f"{name} Model Validation Perplexity: {ppl:.4f}")

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.4f} seconds")
