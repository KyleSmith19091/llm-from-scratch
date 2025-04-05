import os
import json

# load dataset that is valid json
def load_json_data_file(file_name):
    file_path = f"./data/{file_name}"
    if not os.path.exists(file_path):
        raise f"error data file {file_path} does not exist"

    with open(file_path, "r") as file:
        data = json.load(file)
    return data