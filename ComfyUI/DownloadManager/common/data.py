import os
import json
from .config import cfg

def load_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
msst_model_data = load_json('./data/msst_model_map.json')
vr_model_data = load_json('./data/vr_model_map.json')
models_info = load_json('./data/models_info.json')

port = cfg.get(cfg.aria2_port)
ARIA2_RPC_URL = f"http://localhost:{port}/jsonrpc"
ARIA2_RPC_SECRET = cfg.get(cfg.aria2_secret)
HF_ENDPOINT = cfg.get(cfg.hf_endpoint)