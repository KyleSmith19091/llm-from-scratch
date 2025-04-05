import torch
from torch.utils.data import Dataset
from utils.prompt_formatting import format_prompt

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        self.data = data
        self.encoded_texts = []
        for entry in data:
            header = format_prompt(entry) 
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = header + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def collate(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",  
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] 

        padded = (
            # python syntax magic here, constructing an array of pad_token_id with the difference between max length and length of item
            # Max = 5
            # len(new_item) = 3 
            # [1,2,3] + [1,1]
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        
        # remove padded token
        inputs = torch.tensor(padded[:-1])

        # target is the input + 1
        targets = torch.tensor(padded[1:])

        # preapre mask condition
        mask = targets == pad_token_id

        #  returns indices of all non-zero elements i.e. all elements that equal the pad_token_id 
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            # set the values at the indices to ignore_index except the first occurrence
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
