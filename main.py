import tiktoken
import torch
from torch.utils.data import DataLoader
from config import (
    BASE_CONFIG, 
    CHOOSE_MODEL, 
    FINE_TUNING_DATASET_FILE_NAME, 
    BATCH_SIZE,
)
from decoding import (
    generate_text_temp_scaling,
    text_to_token_ids,
    token_ids_to_text,
)
from train import train_model
from model import GPTModel
from utils.load_data import load_json_data_file
from utils.load_weights import load_weights_into_gpt
from utils.optim import calc_loss_loader
from utils.weight_download import download_and_load_gpt2
from dataset import (
    collate,
    format_prompt,
    InstructionDataset
)
from functools import partial

def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load fine-tuning dataset
    data = load_json_data_file(FINE_TUNING_DATASET_FILE_NAME)

    # calculate training, tesrting and validation portions
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1) 
    val_portion = len(data) - train_portion - test_portion

    # split data
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    # fix device and allowed_max_length
    customized_collate_fn = partial(
        collate,
        device=device,
        allowed_max_length=BASE_CONFIG['context_length']
    )

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    ) 

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    ) 

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    ) 

    # load and initialise model
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    train_model(
        device,
        tokenizer,
        model,
        train_loader,
        val_loader,
        val_data,
    )    

    # token_ids = generate_text_temp_scaling(
    #     model=model,
    #     idx=text_to_token_ids(input_text, tokenizer),
    #     max_new_tokens=35,
    #     context_size=BASE_CONFIG["context_length"],
    #     temperature=1.4,
    #     eos_id=198
    # )

    # print("------------------------------------")
    # response_text = token_ids_to_text(token_ids, tokenizer)[len(input_text):].strip()
    # print(response_text)

if __name__ == "__main__":
    main()
