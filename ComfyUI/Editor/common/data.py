import json

def load_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

models_info = load_json('./data/models_info.json')