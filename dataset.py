from typing import Any
import torch
from torch.utils.data import Dataset
import pandas as pd
from tiktoken import Encoding

class SpamDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: Encoding, max_length=None, pad_token_id=50256):
        super().__init__()
        
        # read data into memory
        self.data = pd.read_csv(csv_file)

        # tokenize text in dataset
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]            
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            # truncate all text according to the max length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        
    def __getitem__(self, index) -> Any:
        encoded_text = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded_text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
        
    def __len__(self):
        return self.data

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length