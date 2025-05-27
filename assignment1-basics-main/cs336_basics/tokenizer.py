import json
import regex
class tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        merge_list = []
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line_number, line_content in enumerate(f):
                stripped_line = line_content.strip()
                part = stripped_line.split(" ")
                if len(part) == 2:
                    part1 = part[0]
                    part2 = part[1]
                    part1_byte = part1.encode()
                    part2_byte = part2.encode()
                    merge_list.append((part1_byte, part2_byte))
                else:
                    print(f"Warning: Line {line_number + 1} is not in the expected format: '{stripped_line}'")
        return cls(vocab_dict, merge_list, special_tokens)


    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        compiled_pattern = regex.compile(PAT)
        tokens =  compiled_pattern.findall(text)
        for token in tokens:
            tokenbyte = token.encode()
            bytes_list =  [bytes([i]) for i in tokenbyte]
            for i in range(len(bytes_list)):
                for j in self.merges:
                    if j[0] == bytes_list[i] and j[1] == bytes_list[i + 1]:



    def encode_iterable(self, iterable):




    def decode(self, ids: list[int]) -> str:

    