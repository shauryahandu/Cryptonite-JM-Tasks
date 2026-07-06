import re
import time
import collections


class BPETokenizer:
    def __init__(self, num_merges=200, end_of_word_token="</w>"):
        self.num_merges = num_merges
        self.end_of_word_token = end_of_word_token
        self.merges = []
        self.merge_ranks = {}
        self.vocab = set()
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_word_frequencies(self, corpus):
        word_frequencies = collections.defaultdict(int)
        for line in corpus:
            words = line.strip().split()
            for word in words:
                symbols = tuple(list(word) + [self.end_of_word_token])
                word_frequencies[symbols] += 1
        return word_frequencies

    def _get_pair_frequencies(self, word_frequencies):
        pair_frequencies = collections.defaultdict(int)
        for symbols, frequency in word_frequencies.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_frequencies[pair] += frequency
        return pair_frequencies

    def _merge_pair(self, pair, word_frequencies):
        new_word_frequencies = {}
        merged_symbol = pair[0] + pair[1]
        for symbols, frequency in word_frequencies.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word_frequencies[tuple(new_symbols)] = frequency
        return new_word_frequencies

    def train(self, corpus):
        word_frequencies = self._get_word_frequencies(corpus)
        for symbols in word_frequencies:
            self.vocab.update(symbols)

        for merge_index in range(self.num_merges):
            pair_frequencies = self._get_pair_frequencies(word_frequencies)
            if not pair_frequencies:
                break
            best_pair = max(pair_frequencies, key=pair_frequencies.get)
            if pair_frequencies[best_pair] < 2:
                break
            word_frequencies = self._merge_pair(best_pair, word_frequencies)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = merge_index
            merged_symbol = best_pair[0] + best_pair[1]
            self.vocab.add(merged_symbol)

        sorted_vocab = sorted(self.vocab)
        self.token_to_id = {token: index for index, token in enumerate(sorted_vocab)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.token_to_id["<unk>"] = len(self.token_to_id)
        self.id_to_token[self.token_to_id["<unk>"]] = "<unk>"

    def _apply_merges_to_word(self, word):
        symbols = list(word) + [self.end_of_word_token]
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate_pairs = [pair for pair in pairs if pair in self.merge_ranks]
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs, key=lambda pair: self.merge_ranks[pair])
            merged_symbol = best_pair[0] + best_pair[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text):
        token_ids = []
        words = text.strip().split()
        for word in words:
            symbols = self._apply_merges_to_word(word)
            for symbol in symbols:
                token_id = self.token_to_id.get(symbol, self.token_to_id["<unk>"])
                token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(token_id, "<unk>") for token_id in token_ids]
        text = "".join(tokens)
        text = text.replace(self.end_of_word_token, " ")
        return text.strip()

    def vocab_size(self):
        return len(self.token_to_id)


def word_level_tokenize(text):
    return text.strip().split()


def character_level_tokenize(text):
    return list(text)


def compute_compression_ratio(original_length, encoded_length):
    if encoded_length == 0:
        return 0.0
    return original_length / encoded_length


def compute_average_sequence_length(sequences):
    if not sequences:
        return 0.0
    return sum(len(sequence) for sequence in sequences) / len(sequences)


SAMPLE_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks at the quick fox",
    "a quick fox and a lazy dog become friends",
    "the lazy dog sleeps while the quick fox runs",
    "friendship between the fox and the dog grows quickly",
    "low lower lowest new newer newest wide wider widest",
    "the fox is quicker than the dog but the dog is stronger",
    "the forest is full of foxes dogs and other wild animals",
]


if __name__ == "__main__":
    start_time = time.time()

    tokenizer = BPETokenizer(num_merges=100)
    tokenizer.train(SAMPLE_CORPUS)

    print("BPE Tokenizer Summary")
    print(f"Vocabulary Size: {tokenizer.vocab_size()}")
    print(f"Number of Merge Rules: {len(tokenizer.merges)}")
    print(f"Sample Merge Rules: {tokenizer.merges[:10]}")

    bpe_sequences = [tokenizer.encode(line) for line in SAMPLE_CORPUS]
    word_sequences = [word_level_tokenize(line) for line in SAMPLE_CORPUS]
    char_sequences = [character_level_tokenize(line) for line in SAMPLE_CORPUS]

    print("\nAverage Sequence Length Comparison")
    print(f"Word Level: {compute_average_sequence_length(word_sequences):.4f}")
    print(f"Character Level: {compute_average_sequence_length(char_sequences):.4f}")
    print(f"BPE Level: {compute_average_sequence_length(bpe_sequences):.4f}")

    total_characters = sum(len(line) for line in SAMPLE_CORPUS)
    total_bpe_tokens = sum(len(sequence) for sequence in bpe_sequences)
    compression_ratio = compute_compression_ratio(total_characters, total_bpe_tokens)
    print(f"\nCompression Ratio (characters per BPE token): {compression_ratio:.4f}")

    oov_word = "foxhound"
    oov_encoded = tokenizer.encode(oov_word)
    oov_decoded = tokenizer.decode(oov_encoded)
    print("\nOut Of Vocabulary Behaviour")
    print(f"Input Word: {oov_word}")
    print(f"Encoded Ids: {oov_encoded}")
    print(f"Decoded Reconstruction: {oov_decoded}")

    example_sentence = "the quick fox becomes stronger and quicker"
    encoded_example = tokenizer.encode(example_sentence)
    decoded_example = tokenizer.decode(encoded_example)

    print("\nExample Encoding And Decoding")
    print(f"Original Sentence: {example_sentence}")
    print(f"Encoded Ids: {encoded_example}")
    print(f"Decoded Sentence: {decoded_example}")

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.4f} seconds")
