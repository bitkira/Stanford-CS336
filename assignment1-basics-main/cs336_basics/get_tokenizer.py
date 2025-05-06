import regex as re
from collections import defaultdict
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def BPE(input_path: str, vocab_size: int, special_tokens: list[str]):
    ## Usage
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 3, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tokens = re.findall(PAT, chunk)
            print(tokens)

            vocab:dict[int, bytes] = {i:bytes([i]) for i in range(256)}
            merges: list[tuple[int, int], int] = []
            process = [list(i.encode("utf-8")) for i in tokens]
            
            for i in range(vocab_size):
                count = defaultdict(int)
                for j in process:
                    for index1 , index2 in zip(j, j[1:]):
                        count[(index1, index2)] += 1
                pair = max(count, key=count.get)
                #pair = max(count, key=lambda k: (count.get(k), k))
                merges.append(pair)
                index1, index2 = pair
                vocab[i+256] = vocab[index1]+vocab[index2]
                newprocess = []
                for k in process:
                    microprocess = []
                    l = 0
                    while(l<len(k)):
                        if (l<len(k) - 1 and k[l]==pair[0] and k[l+1]==pair[1]):
                            microprocess.append(i+256)
                            l=l+2
                        else:
                            microprocess.append(k[l])
                            l = l+1
                    newprocess.append(microprocess)
                process = newprocess
            
            return vocab, merges
            