"""
Built off of Tiktoken educational implementation 
https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
and this pytorch implementation of karpathy's minbpe
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
import multiprocessing as mp
import glob
from functools import partial, lru_cache

import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node

# Multi-GPU setup with torchrun
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
print(f"Running with {world_size} GPU(s)")
assert torch.cuda.is_available()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
# Initialize distributed process group
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.


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
    def train(data_dir: str, vocab_size: int, pat_str: str, k: int = 256, np_dtype: np.dtype = np.int16):
        """Train a BPE tokeniser using data loaded from shards."""
        
        # Determine torch int_type based on vocab_size
        int_type = torch.int16 if vocab_size <= (2**15)-1 else torch.int32 
        assert vocab_size <= (2**31)-1, f"Vocab size {vocab_size} too large"
        # Ensure torch and numpy types are consistent
        assert (int_type == torch.int16 and np_dtype == np.int16) or \
               (int_type == torch.int32 and np_dtype == np.int32), \
               f"Mismatch between torch type {int_type} and numpy type {np_dtype}"

        # Load data from shards - pass np_dtype for reading, int_type for final tensor
        ids = load_from_shards(data_dir, np_dtype, int_type) 

        # Pass the loaded tensor and pat_str (for demo) to bpe_train
        mergeable_ranks = bpe_train(
            ids=ids, 
            vocab_size=vocab_size, 
            pat_str=pat_str, # Pass pat_str for demo purposes
            k=k,
            int_type=int_type # Pass torch int_type
            ) 
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


@lru_cache
def nat2int(num: int):
    """
    converts natural numbers to integer counterparts for use in
    efficiently utilizing signed int datatypes
    (0, 1, 2, 3,...) -> (0, -1, 1, -2, 2,...)
    """
    if num % 2 == 0:  # even numbers map to positive
        return num // 2
    else: # odd numbers map to negative
        return -(num + 1) // 2
  

@lru_cache 
def int2nat(num: int):
    """
    converts integer numbers to natural counterparts for use in
    efficiently utilizing signed int datatypes
    (0, -1, 1, -2, 2,...) -> (0, 1, 2, 3,...)
    """
    if num >= 0:  # positive numbers map back to even
        return 2 * num
    else:  # negative numbers map back to odd
        return -2 * num - 1


def bpe_train(
    ids: torch.Tensor, vocab_size: int, pat_str: str, k: int = 256, int_type: torch.dtype = torch.int16
) -> dict[bytes, int]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    
    # ranks maps bytes -> natural IDs (0, 1, 2...)
    ranks = {bytes([i]): i for i in range(2**8)}
    # id_to_bytes maps natural IDs -> bytes
    id_to_bytes = {i: bytes([i]) for i in range(2**8)} 
    
    # set indicator tokens for merging ops
    type_info = torch.iinfo(int_type)
    SEPARATOR_TOKEN = type_info.min
    REMOVE_TOKEN = type_info.max
    
    if master_process:
        # Initialize demo text tokens outside the loop to track changes across iterations
        demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                    f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                    f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
        demo_bytes = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]
        
    # Now, use our data to figure out which merges we should make
    progress_bar = tqdm(total=vocab_size - 256, unit="merges", disable=not master_process)
    for j in range(256, vocab_size):
        if ids.numel() < 2:
             print(f"Rank {rank}: Not enough elements ({ids.numel()}) to form pairs. Stopping merges early.")
             break
             
        pairs = torch.stack((ids[:-1], ids[1:]), dim=0) # (2, words_in_data * (avg_word_len + 1))
        unique, counts = torch.unique(pairs, return_counts=True, dim=1)
            # shapes (2, very_long) and (very_long)
            # where very_long < words_in_data * (avg_word_len + 1)
        
        # use separator token between words to ensure we follow regex
        valid_mask = torch.all(unique != SEPARATOR_TOKEN, dim=0) # (very_long)
        unique = unique[:, valid_mask] # (2, very_long)
        counts = counts[valid_mask] # (very_long)

        # select top k pairs to go into consideration
        counts, sort_idx = torch.sort(counts, descending=True) # (very_long) and (very_long)
        pairs_idx = sort_idx[:k] # shape (k)
        most_common_pairs_local = unique[:, pairs_idx] # (2, k)
        counts_local = counts[:k]# (k)
        
        # communicate between GPUs
        most_common_pairs_global = torch.zeros((2, k * world_size), dtype=torch.float32, device=device)
        counts_global = torch.zeros(k * world_size, dtype=torch.float32, device=device)
        most_common_pairs_global[:, rank * k : (rank + 1) * k] = most_common_pairs_local.to(torch.float32)
        counts_global[rank * k : (rank + 1) * k] = counts_local.to(torch.float32)
        dist.all_reduce(most_common_pairs_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts_global, op=dist.ReduceOp.SUM)

        # get unique pairs and their counts from the combined data
        unique_pairs, inverse_indices = torch.unique(most_common_pairs_global.t(), dim=0, return_inverse=True)

        # Sum the counts for each unique pair
        sum_counts = torch.zeros(unique_pairs.size(0), dtype=torch.float, device=device)
        sum_counts.scatter_add_(0, inverse_indices, counts_global.float())

        # Count occurrences of each unique pair
        pair_occurrences = torch.bincount(inverse_indices)

        # Find the maximum occurrence count
        max_occurrence = torch.max(pair_occurrences)

        # Create a mask for pairs with the maximum occurrence count
        max_occurrence_mask = (pair_occurrences == max_occurrence)

        # Filter to only consider pairs with the maximum occurrence count
        filtered_sum_counts = sum_counts[max_occurrence_mask]
        filtered_unique_pairs = unique_pairs[max_occurrence_mask]

        # Find the pair with the largest count among the filtered pairs
        max_index = torch.argmax(filtered_sum_counts)
        best_pair_signed = filtered_unique_pairs[max_index].cpu().numpy() # (2)
            
        # --- Map chosen pair (signed IDs) back to bytes using id_to_bytes ---
        best_pair_nat = [int2nat(bp_signed) for bp_signed in best_pair_signed]
        
        try:
            # Direct lookup using the reverse map
            bytes_0 = id_to_bytes[best_pair_nat[0]]
            bytes_1 = id_to_bytes[best_pair_nat[1]]
        except KeyError as e:
             print(f"Error: Could not find bytes for natural ID {e}. This shouldn't happen.")
             print(f"Attempted lookup for pair: signed={best_pair_signed}, natural={best_pair_nat}")
             continue # Skip this merge iteration

        token_bytes = bytes_0 + bytes_1
        new_token_id_nat = len(ranks) # Natural ID for the new token (0, 1, 2...)
        new_token_id_signed = nat2int(new_token_id_nat) # Signed ID for tensor ops

        # Add the new token to both maps
        ranks[token_bytes] = new_token_id_nat 
        id_to_bytes[new_token_id_nat] = token_bytes # Update reverse map

        # Now merge that most common pair in all the words
        pair_mask = (pairs[0] == best_pair_signed[0]) & (pairs[1] == best_pair_signed[1]) 
        ids[:-1][pair_mask] = new_token_id_signed
        ids[1:][pair_mask] = REMOVE_TOKEN
        keep_mask = (ids != REMOVE_TOKEN)
        ids = ids[keep_mask]

        if master_process:
            progress_bar.update(1)

            # Also apply the same merge to our demo text
            demo_bytes = merge_bytes(demo_bytes, (bytes_0, bytes_1), token_bytes)

            # See the intermediate merges play out!
            if j % 1000 == 0 or j in [256, vocab_size - 1]:
                print(f"\nThe most common pair {best_pair_nat[0]} + {best_pair_nat[1]} "
                        f"which makes {token_bytes} our {len(ranks)}th token")
                # Flatten the demo words into a single list of tokens for visualization
                flattened_demo_bytes = [token for word in demo_bytes for token in word]
                visualise_tokens(flattened_demo_bytes)
    
    if master_process: 
        progress_bar.close()
        print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    return ranks


def load_tokenizer(tokenizer_path):
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


# NOTE: not currently utilized; here as an example
def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


# This function might be called by multiple processes, so handle initialization carefully.
def tokenize_doc(doc):
    # Convert text directly to byte values
    tokens = np.array(list(doc["text"].encode("utf-8")))
    # Byte values are 0-255, fitting within uint8, but downstream will expect uint16
    assert (0 <= tokens).all() and (tokens < 256).all(), "Byte values out of range 0-255"
    tokens_uint8 = tokens.astype(np.uint8)
    return tokens_uint8


def tokenize_worker(doc_ids, np_dtype):
    """Worker function to convert list of signed IDs to numpy array of correct signed type."""
    # Convert list directly to the specified numpy signed dtype
    tokens = np.array(doc_ids, dtype=np_dtype)
    return tokens


def fetch_fineweb_data(vocab_size: int, pat_str: str, max_chars: int = 2**22, shard_size: int = 10**8):
    """Fetch data, tokenize with separators using correct signed types, and save shards."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine int_type and numpy equivalent based on vocab_size
    if vocab_size <= (2**15) - 1:
        int_type = torch.int16
        np_dtype = np.int16
        SEPARATOR_TOKEN = torch.iinfo(torch.int16).min # Usually -32768
    elif vocab_size <= (2**31) - 1:
        int_type = torch.int32
        np_dtype = np.int32
        SEPARATOR_TOKEN = torch.iinfo(torch.int32).min # Usually -2147483648
    else:
         raise ValueError(f"vocab_size {vocab_size} too large for supported types")

    print(f"Using {int_type} (numpy: {np_dtype}) for tokens. SEPARATOR_TOKEN={SEPARATOR_TOKEN}")

    # File naming pattern for byte shards
    shard_pattern = os.path.join(data_dir, f"fineweb_{np_dtype.__name__}_" + "{index:06d}.bin")

    # --- Master process handles downloading and processing ---
    if rank == 0:
        # TODO: Add logic to check if shards totaling >= target_bytes already exist
        # TODO: Add logic to clean up old/incomplete shards if resuming

        print(f"Master process downloading and tokenizing FineWeb data (target: {max_chars:,} chars)...")
        
        dataset = load_dataset("HuggingFaceFW/fineweb", 
                              name="sample-100BT",
                              split="train", 
                              streaming=True)
        
        nprocs = max(1, os.cpu_count() - 2)
        print(f"Using {nprocs} processes for tokenization.")
        
        total_bytes_written = 0
        shard_index = 0
        current_shard_bytes = 0
        current_shard_fh = None 

        # --- Function to open the next shard file ---
        def open_next_shard():
            nonlocal current_shard_fh, shard_index, current_shard_bytes
            if current_shard_fh is not None:
                current_shard_fh.close()
            shard_filename = shard_pattern.format(index=shard_index)
            print(f"Opening shard file: {shard_filename}")
            current_shard_fh = open(shard_filename, "wb")
            current_shard_bytes = 0
            shard_index += 1 # Increment index for the *next* shard

        # Initial byte ranks (0-255)
        ranks = {bytes([i]): i for i in range(2**8)}

        # Pass np_dtype to the worker function via partial or modify worker
        tokenize_partial = partial(tokenize_worker, np_dtype=np_dtype) 

        with mp.Pool(nprocs) as pool, tqdm(total=max_chars, unit="chars", desc="Processing chars") as pbar:
            chars_processed_so_far = 0
            doc_iterator = iter(dataset)
            
            open_next_shard() # Open the very first shard file

            while chars_processed_so_far < max_chars:
                # Fetch a batch of documents to reduce overhead
                batch_size = nprocs * 16 # Heuristic batch size
                docs_batch_for_pool = [] # Prepare list of lists for the pool
                batch_char_count = 0
                try:
                    for _ in range(batch_size):
                        doc = next(doc_iterator)["text"]
                        doc_len = len(doc)

                        at_end = chars_processed_so_far + batch_char_count + doc_len >= max_chars
                        current_doc_char_limit = doc_len
                        
                        if at_end:
                            # Add partial document if needed to reach max_chars exactly
                            remaining_chars = max_chars - (chars_processed_so_far + batch_char_count)
                            if remaining_chars <= 0: # Already reached or exceeded max_chars
                                chars_processed_so_far = max_chars # Ensure loop terminates
                                break # Break inner loop - don't process this doc
                            doc = doc[:remaining_chars]
                            current_doc_char_limit = remaining_chars
                            batch_char_count += remaining_chars
                        else:
                            batch_char_count += doc_len

                        # Splinter up our data into lists of bytes based on the regex pattern
                        words: list[list[bytes]] = [
                            [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, doc)
                        ]
                        # Create a list to store numeric token IDs for tensor operations
                        # Initially, these are just byte values (0-255)
                        byte_ids_lofl = [[ranks[b] for b in word] for word in words]
                        # convert from (0,1,2,3...) to (0,-1,1,-2,2,...) to saturate int dtype (pytorch doesn't support uint well)
                        ids_lofl = [[nat2int(num) for num in sublist] for sublist in byte_ids_lofl] 
                        # Use the correct SEPARATOR_TOKEN value
                        ids = list(chain.from_iterable(word + [SEPARATOR_TOKEN] for word in ids_lofl)) 
                        # Append the list of signed IDs to the batch for the pool
                        docs_batch_for_pool.append(ids)
                        
                        if at_end:
                            chars_processed_so_far = max_chars # Ensure loop terminates correctly
                            break # Stop fetching more docs for this batch
                            
                except StopIteration:
                    # Dataset finished before reaching max_chars
                    chars_processed_so_far += batch_char_count
                    break # Break outer loop

                if not docs_batch_for_pool:
                    if chars_processed_so_far >= max_chars: # Check if loop should terminate
                         break 
                    # This case might occur if the first doc itself is > max_chars
                    print("Warning: No documents processed in a batch cycle.")
                    continue # Or break, depending on desired behavior

                # Process the batch using imap_unordered and write results to shards
                try:
                    for tokens_array in pool.imap_unordered(tokenize_partial, docs_batch_for_pool):
                        bytes_to_write = tokens_array.nbytes
                        
                        # Write to current shard, potentially splitting across shards
                        while bytes_to_write > 0:
                            space_left_in_shard = shard_size - current_shard_bytes
                            
                            bytes_in_chunk = min(bytes_to_write, space_left_in_shard)
                            elements_in_chunk = bytes_in_chunk // tokens_array.itemsize

                            if elements_in_chunk > 0:
                                # Get the slice of the numpy array corresponding to the chunk
                                data_chunk = tokens_array[:elements_in_chunk]
                                current_shard_fh.write(data_chunk.tobytes())
                                current_shard_bytes += data_chunk.nbytes
                                total_bytes_written += data_chunk.nbytes
                                
                                # Prepare the remainder
                                tokens_array = tokens_array[elements_in_chunk:]
                                bytes_to_write -= data_chunk.nbytes
                            
                            # Check if the shard is full or if we finished writing this array
                            if current_shard_bytes >= shard_size or bytes_to_write == 0:
                                if current_shard_bytes >= shard_size:
                                     open_next_shard()
                                # If bytes_to_write > 0, the loop continues with the new shard

                except Exception as e:
                    print(f"Error during tokenization/writing: {e}")
                    # Decide how to handle errors, e.g., stop or log and continue
                    if current_shard_fh: current_shard_fh.close()
                    raise 

                # Update progress based on characters *intended* for this batch
                # Note: Actual chars processed might be slightly less if ended early
                pbar.update(batch_char_count) 

        # Close the last shard file if it's open
        if current_shard_fh is not None:
            current_shard_fh.close()
            print(f"Closing final shard file.")

        print(f"Master process finished targeting {max_chars:,} characters.")
        print(f"Total bytes written to shards: {total_bytes_written:,}")
        # TODO: Signal completion or write metadata about shards

    # --- All processes wait here ---
    dist.barrier()
    print(f"Rank {rank} finished waiting for data preparation.")
    
    # Return the directory and the determined numpy dtype for loading
    return data_dir, np_dtype 


def load_from_shards(data_dir: str, np_dtype: np.dtype, int_type: torch.dtype, chunk_size: int = 10 * 1024 * 1024) -> torch.Tensor:
    """Loads signed token data incrementally from shards."""
    # Use the dtype name in the pattern to find the correct files
    shard_pattern = os.path.join(data_dir, f"fineweb_{np_dtype.__name__}_*.bin") 
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found matching {shard_pattern}")

    # Distribute shards among ranks
    files_for_rank = shard_files[rank::world_size]

    if not files_for_rank:
        raise ValueError(f"Rank {rank} has no shards assigned.")

    print(f"Rank {rank} loading {len(files_for_rank)} shards ({np_dtype.__name__}) in chunks of {chunk_size // 1024 // 1024} MiB...")

    all_tensor_chunks = [] # Store tensor chunks from all files for this rank
    total_bytes = 0
    
    for filename in tqdm(files_for_rank, desc=f"Rank {rank} loading shards"):
        try:
            # Ensure chunk size is a multiple of item size for clean reads
            item_size = np.dtype(np_dtype).itemsize
            adjusted_chunk_size = (chunk_size // item_size) * item_size
            if adjusted_chunk_size == 0: # Handle very small chunk sizes
                 adjusted_chunk_size = item_size

            with open(filename, "rb") as f:
                while True:
                    chunk_bytes = f.read(adjusted_chunk_size)
                    if not chunk_bytes:
                        break # End of file

                    # Convert bytes directly using the correct signed numpy dtype
                    token_data_np = np.frombuffer(chunk_bytes, dtype=np_dtype)
                    total_tokens += len(token_data_np)
                    
                    # Convert numpy array to torch tensor directly on the GPU with the target torch int_type
                    tensor_chunk = torch.from_numpy(token_data_np).to(device=device, dtype=int_type)
                    all_tensor_chunks.append(tensor_chunk)
                    
                    # Optional: Add a small sleep or yield if I/O is overwhelming other processes,
                    # but usually direct reading is fine.
                    # time.sleep(0.001) 

        except Exception as e:
            print(f"Rank {rank} error reading {filename}: {e}")
            # Decide how to handle errors, maybe raise or return partial data
            raise

    if not all_tensor_chunks:
        raise ValueError(f"Rank {rank} - No data loaded.")
        
    # Concatenate all tensor chunks ONCE
    ids = torch.cat(all_tensor_chunks)
    all_tensor_chunks = [] 
    # total_bytes = total_tokens * item_size # Calculate total bytes loaded
    print(f"Rank {rank} loaded {total_tokens:,} tokens (including separators) -> tensor shape {ids.shape} ({ids.dtype})")

    return ids


def train_simple_encoding(sample_size: int, vocab_size: int, k: int = 256):
    """
    Train a custom BPE tokenizer using FineWeb data.
    
    Args:
        sample_size: maximum number of characters to include in data
        vocab_size: Size of the vocabulary to train
        k: number of pairs to compare across GPUs
    
    Returns:
        The trained tokenizer
    """
    #gpt2_pattern = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    gpt4_pattern = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    # Pass vocab_size to fetch_fineweb_data
    data_dir, np_dtype = fetch_fineweb_data(vocab_size=vocab_size, pat_str=gpt4_pattern, max_chars=sample_size)
    
    # Pass data_dir, vocab_size, pattern, np_dtype (for loading) to train
    enc = SimpleBytePairEncoding.train(
        data_dir=data_dir, 
        vocab_size=vocab_size, 
        pat_str=gpt4_pattern, 
        k=k, 
        np_dtype=np_dtype
        )
    
    # Test the tokenizer with a simple example
    test_str = f"hello world rank={rank}"
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
    full_filename = f"tokenizers/multiple_GPUs_v{vocab_size}_n{sample_size}.pkl"
    
    # Prepare the tokenizer data
    tokenizer_data = {
        "pat_str": enc.pat_str,
        "mergeable_ranks": enc.mergeable_ranks
    }
    
    # Save the tokenizer data
    with open(full_filename, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    
    print(f"Tokenizer saved to {full_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("-n", "--samples", type=int, default=world_size * (2**27), 
        help=(f"Maximum number of text characters to use (across all GPUs)"
            f" for training (2^27 should fit on single GPU with 8gb of VRAM)"))
    parser.add_argument("-v", "--vocabsize", type=int, default=65534, 
        help="Size of the vocabulary to train (default 65,534 to saturate int16)")
    args = parser.parse_args()

    # this is the number of top-k unique pairs set to be communicated between GPUs
    k = 256 # set heuristically, shouldn't be very important

    # Train the tokenizer
    enc = train_simple_encoding(
        sample_size=args.samples,
        vocab_size=args.vocabsize,
        k=k
    )
    
    # Save the tokenizer
    if master_process:
        save_tokenizer(enc, args.vocabsize, args.samples)

        # Demonstrate the tokenizer usage
        print("\nDemonstrating tokenizer usage:")
        
        # Use the tokenizer with the known vocab size and sample size
        tokenizer_filename = f"multiple_GPUs_v{args.vocabsize}_n{args.samples}.pkl"
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

    dist.destroy_process_group()
        
    