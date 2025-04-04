import tiktoken
from dataset import SpamDataset
from model import GPTModel
from weight_download import download_and_load_gpt2 
from load_weights import load_weights_into_gpt
from config import CHOOSE_MODEL, BASE_CONFIG
from decoding import generate_text_greedy, text_to_token_ids, token_ids_to_text, generate_text_temp_scaling

def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    model.eval() # disable dropout
    load_weights_into_gpt(model, params)

    text_1 = "The capital of Texas is"
    token_ids = generate_text_temp_scaling(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=40,
        context_size=BASE_CONFIG["context_length"],
        temperature=1.4,
        eos_id=198
    )

    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
