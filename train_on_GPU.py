"""
expanded upon from this pytorch implementation of karpathy's minbpe which didn't support regex
https://github.com/kuprel/minbpe-pytorch/tree/main
"""
from __future__ import annotations
import collections
import regex
import tiktoken
import os
import shutil
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pickle
from itertools import chain

# Single GPU setup
import torch
assert torch.cuda.is_available()
device = "cuda"

def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # demo the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")


class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(pat_str)

    def bpe_encode(self, mergeable_ranks: dict[bytes, int], input: bytes) -> list[int]:
        parts = [bytes([b]) for b in input]
        while True:
            # Iterate over all pairs and find the pair we want to merge the most
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank

            # If there were no pairs we could merge, we're done!
            if min_rank is None:
                break
            assert min_idx is not None

            # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

        tokens = [mergeable_ranks[part] for part in parts]
        return tokens

    def encode(self, text: str, demo: bool = False) -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = self.bpe_encode(self.mergeable_ranks, word_bytes)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str, demo: bool = False, k: int = 256):
        """Train a BPE tokeniser on some data!"""
        mergeable_ranks = bpe_train(data=training_data, vocab_size=vocab_size, pat_str=pat_str, demo=demo, k=k)
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

    @staticmethod
    def from_tiktoken(encoding):
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)
        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks
        )


def merge_bytes(words, most_common_pair, token_bytes):
    new_words = []
    for word in words:
        new_word = []
        i = 0
        while i < len(word) - 1:
            if (word[i], word[i + 1]) == most_common_pair:
                # We found our pair! Merge it
                new_word.append(token_bytes)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if i == len(word) - 1:
            new_word.append(word[i])
        new_words.append(new_word)
    return new_words


def bpe_train(
    data: str, vocab_size: int, pat_str: str, demo: bool = False, k: int = 256
) -> dict[bytes, int]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i
    
    # choose efficient data type
    int_type = torch.int16 if vocab_size <= (2**15) else torch.int32
    assert vocab_size <= (2**31), f"bro why you making such a big tokenizer? {vocab_size}"
    # set indicator tokens for merging ops
    SEPARATOR_TOKEN = -1 
    REMOVE_TOKEN = -2

    # Splinter up our data into lists of bytes
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, data)
    ]
    # Create a list to store numeric token IDs for tensor operations
    # Initially, these are just byte values (0-255)
    ids_lofl = [[ranks[b] for b in word] for word in words]
    # turn data into parseable tensor - using the token IDs instead of raw bytes
    ids = torch.tensor(
        list(chain.from_iterable(word + [SEPARATOR_TOKEN] for word in ids_lofl))[:-1], 
        dtype=int_type, device=device)
        # shape (words_in_data * (avg_word_len + 1))\

    # Initialize demo text tokens outside the loop to track changes across iterations
    demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
    demo_bytes = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    # Now, use our data to figure out which merges we should make
    for j in tqdm(range(256, vocab_size), unit="merges"):
        # find frequency of all pairs
        pairs = torch.stack((ids[:-1], ids[1:]), dim=0) # (2, words_in_data * (avg_word_len + 1))
        unique, counts = torch.unique(pairs, return_counts=True, dim=1)
            # shapes (2, very_long) and (very_long)
            # where very_long < words_in_data * (avg_word_len + 1)
        
        # use separator token between words to ensure we follow regex
        valid_mask = torch.all(unique != SEPARATOR_TOKEN, dim=0) # (very_long)
        unique = unique[:, valid_mask] # (2, very_long)
        counts = counts[valid_mask] # (very_long)

        pair_idx = torch.argmax(counts) # (1)
        best_pair = unique[:, pair_idx].cpu().numpy() # (2)
            
        # Map token IDs back to the corresponding byte sequences
        # Using the dictionary in reverse to get the bytes corresponding to these IDs
        best_bytes = [None, None]
        best_pair_0 = best_pair[0]
        best_pair_1 = best_pair[1]
        for bytes_token, id_token in ranks.items():
            if id_token == best_pair_0:
                best_bytes[0] = bytes_token
            if id_token == best_pair_1:
                best_bytes[1] = bytes_token
        token_bytes = best_bytes[0] + best_bytes[1]
        new_token_id = len(ranks)
        # Add the new token!
        ranks[token_bytes] = new_token_id

        # Now merge that most common pair in all the words
        pair_mask = (pairs[0] == best_pair[0]) & (pairs[1] == best_pair[1]) 
        ids[:-1][pair_mask] = new_token_id
        ids[1:][pair_mask] = REMOVE_TOKEN
        keep_mask = (ids != REMOVE_TOKEN)
        ids = ids[keep_mask]

        # Also apply the same merge to our demo text
        demo_bytes = merge_bytes(demo_bytes, tuple(best_bytes), token_bytes)

        # See the intermediate merges play out!
        if j % 100 == 0 or j in [256, vocab_size - 1]:
            print(f"\nThe most common pair {best_pair[0]} + {best_pair[1]} "
                    f"which makes {token_bytes} our {len(ranks)}th token")
            # Flatten the demo words into a single list of tokens for visualization
            flattened_demo_tokens = [token for word in demo_bytes for token in word]
            visualise_tokens(flattened_demo_tokens)
    
    #print(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            #f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    return ranks


def fetch_fineweb_data(max_chars: int):
    """Fetch data from FineWeb dataset for tokenizer training"""
    # Create a local cache directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Check for existing files that meet the size requirement
    existing_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt") and file.startswith("tokenizer_training_data_"):
            try:
                # Extract size from filename like 'tokenizer_training_data_1000.txt'
                file_size = int(file.split("_")[-1].split(".")[0])
                existing_files.append((file, file_size))
            except (ValueError, IndexError):
                # Ignore files with unexpected naming format
                continue

    if existing_files:
        # Find suitable existing files (>= max_chars)
        suitable_files = [(f, s) for f, s in existing_files if s >= max_chars]

        if suitable_files:
            # Use the smallest file that meets our requirements
            suitable_files.sort(key=lambda x: x[1])
            filename, _ = suitable_files[0] # Get filename from the tuple
            filepath = os.path.join(data_dir, filename) # Construct the full path
            print(f"Using existing data file: {filepath}") # Inform the user
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(max_chars) # Read up to max_chars from the file
            return content # Return the actual file content
    
    # Clean up smaller files
    for file in os.listdir(data_dir):
        if file.startswith("tokenizer_training_data_") and file.endswith(".txt"):
            try:
                file_size = int(file.split("_")[-1].split(".")[0])
                if file_size < max_chars:
                    print(f"Removing smaller existing data file: {file} ({file_size:,} chars < {max_chars:,} chars)")
                    os.remove(os.path.join(data_dir, file))
            except ValueError:
                continue

    # Download new data
    new_file_name = f"tokenizer_training_data_{max_chars}.txt"
    local_data_path = os.path.join(data_dir, new_file_name)
    print(f"Downloading FineWeb data to {local_data_path}...")

    dataset = load_dataset("HuggingFaceFW/fineweb",
                            name="sample-10BT",
                            split="train",
                            streaming=True)

    text_data = []
    doc_lengths = []
    tot_len = 0
    for item in dataset:
        text_data.append(item["text"])
        doc_lengths.append(len(item["text"]))
        tot_len += len(item["text"])
        if tot_len >= max_chars:
            break

    # Show statistics
    print(f"\nDataset Statistics:"
        f"\nTotal documents: {len(text_data)}"
        f"\nTotal characters: {sum(doc_lengths):,}"
        f"\nAverage document length: {np.mean(doc_lengths):.1f} characters"
        f"\nMedian document length: {np.median(doc_lengths):.1f} characters"
        f"\nShortest document: {min(doc_lengths)} characters"
        f"\nLongest document: {max(doc_lengths):,} characters"
        f"\nStandard deviation: {np.std(doc_lengths):.1f} characters")
        
    # Save the combined text to a file
    final_text = "\n".join(text_data)[:max_chars]
    with open(local_data_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
        
    return final_text


def train_simple_encoding(sample_size: int, vocab_size: int):
    """
    Train a custom BPE tokenizer using FineWeb data.
    
    Args:
        sample_size: maximum number of characters to include in data
        vocab_size: Size of the vocabulary to train
    
    Returns:
        The trained tokenizer
    """
    data = fetch_fineweb_data(max_chars=sample_size)
    
    #gpt2_pattern = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    gpt4_pattern = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    enc = SimpleBytePairEncoding.train(data, vocab_size=vocab_size, pat_str=gpt4_pattern)
    
    # Test the tokenizer with a simple example
    test_str = f"hello world"
    tokens = enc.encode(test_str)
    # Verify encoding-decoding roundtrips correctly
    decoded = enc.decode(tokens)
    assert decoded == test_str, f"Decoding failed: expected '{test_str}' but got '{decoded}'"
    decoded_bytes = enc.decode_bytes(tokens)
    assert decoded_bytes == test_str.encode('utf-8'), \
        f"Bytes decoding failed: expected {test_str.encode('utf-8')} but got {decoded_bytes}"
    
    return enc


def save_tokenizer(enc, vocab_size, sample_size):
    """Save the tokenizer for later use"""
    # Ensure the directory exists
    os.makedirs('tokenizers', exist_ok=True)
    
    # Construct the filename
    full_filename = f"tokenizers/GPU_v{vocab_size}_n{sample_size}.pkl"
    
    # Prepare the tokenizer data
    tokenizer_data = {
        "pat_str": enc.pat_str,
        "mergeable_ranks": enc.mergeable_ranks,
    }
    
    # Save the tokenizer data
    with open(full_filename, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    
    print(f"Tokenizer saved to {full_filename}")


def load_tokenizer(tokenizer_path):
    """this function can be imported by other .py files to use this tokenizer"""
    tokenizer_config = pickle.load(open(tokenizer_path, 'rb'))
    enc = tiktoken.Encoding(
        name=tokenizer_path.split('/')[-1][:-4], # Use filename without extension as name
        pat_str=tokenizer_config['pat_str'],
        mergeable_ranks=tokenizer_config['mergeable_ranks'],
        special_tokens={
            "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
        }
    )
    eot = enc._special_tokens['<|endoftext|>']
    return enc, eot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("-n", "--samples", type=int, default=2**20, 
        help=(f"Maximum number of text characters to use for training"))
    parser.add_argument("-v", "--vocabsize", type=int, default=1000, 
        help="Size of the vocabulary to train (minus <|endoftext|>)")
    args = parser.parse_args()

    # Train the tokenizer
    enc = train_simple_encoding(sample_size=args.samples, vocab_size=args.vocabsize)
    
    # Save the tokenizer
    save_tokenizer(enc, args.vocabsize, args.samples)

    # Demonstrate the tokenizer usage
    print("\nDemonstrating tokenizer usage:")
    
    # Use the tokenizer with the known vocab size and sample size
    tokenizer_filename = f"GPU_v{args.vocabsize}_n{args.samples}.pkl"
    data_dir = os.path.join(os.path.dirname(__file__), "tokenizers")
    tokenizer_path = os.path.join(data_dir, tokenizer_filename)
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer file {tokenizer_filename} not found.")
    else:
        print(f"Loading tokenizer: {tokenizer_filename}")
        
        # Load the tokenizer
        enc, eot = load_tokenizer(tokenizer_path)
        
        # Create a sample text
        sample_text = "This is a test of the custom tokenizer."
        
        # Encode the text
        tokens = enc.encode(sample_text)
        
        # Print results
        print(f"\nSample text: '{sample_text}'")
        print(f"Encoded tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        
        # Decode back to text
        decoded_text = enc.decode(tokens)
        print(f"Decoded text: '{decoded_text}'")
        
        # Show some token-to-text mappings
        print("\nSome token-to-text mappings:")
        for i, token in enumerate(tokens[:10]):  # Show first 10 tokens
            print(f"Token {token} → '{enc.decode([token])}'")
    