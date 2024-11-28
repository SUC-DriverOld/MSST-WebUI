"""
{
    "uid": null,
    "model_name": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "model_type": "bs_roformer",
    "path": "./pretrain/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "config_path": "configs/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.yaml",
    "input": [
        "input"
    ],
    "output": [
        "vocals",
        "instrumental"
    ],
    "parameter": [
        {
            "parameter": "batch_size",
            "type": "int",
            "default_value": 1,
            "max_value": 100,
            "min_value": 1,
            "current_value": 1
        },
        {
            "parameter": "dim_t",
            "type": "int",
            "default_value": 901,
            "max_value": 10000,
            "min_value": 1,
            "current_value": 901
        },
        {
            "parameter": "num_overlap",
            "type": "int",
            "default_value": 4,
            "max_value": 100,
            "min_value": 1,
            "current_value": 4
        }
    ],
    "bool": [
        {
            "parameter": "use_cpu",
            "default_value": false,
            "current_value": false
        },
        {
            "parameter": "use_tta",
            "default_value": false,
            "current_value": false
        }
    ],
    "down_stream_nodes": [],
    "up_stream_node": null,
    "output_format": "wav",
    "scene_pos": [
        0,
        0
    ],
    "input_path": null,
    "output_path": null
}
"""
import os, sys
sys.path.append(os.getcwd())
from inference.comfy_infer import ComfyMSST
from ml_collections import ConfigDict
from omegaconf import OmegaConf
import yaml

def msst_inference(node_dict, logger=None):
    config_path = node_dict["config_path"]
    model_type = node_dict["model_type"]
    with open(config_path) as f:
        if model_type == 'htdemucs':
            config = OmegaConf.load(config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            
    for parameter in node_dict["parameter"]:
        if parameter["parameter"] == "batch_size":
            batch_size = parameter["current_value"]
        elif parameter["parameter"] == "dim_t":
            dim_t = parameter["current_value"]
        elif parameter["parameter"] == "num_overlap":
            num_overlap = parameter["current_value"]
            
    for bool_parameter in node_dict["bool"]:
        if bool_parameter["parameter"] == "use_cpu":
            use_cpu = bool_parameter["current_value"]
        elif bool_parameter["parameter"] == "use_tta":
            use_tta = bool_parameter["current_value"]
        elif bool_parameter["parameter"] == "normalize":
            normalize = bool_parameter["current_value"]           
            
    if config.inference.get('batch_size'):
        config.inference['batch_size'] = int(batch_size)
    if config.inference.get('dim_t'):
        config.inference['dim_t'] = int(dim_t)
    if config.inference.get('num_overlap'):
        config.inference['num_overlap'] = int(num_overlap)
    if config.inference.get('normalize'):
        config.inference['normalize'] = normalize
        
    with open(config_path, 'w') as f:
        if model_type == 'htdemucs':
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config.to_dict()
        yaml.dump(config_dict, f)
        
    separator = ComfyMSST(
        model_type=model_type,
        model_path=node_dict["path"],
        config_path=config_path,
        output_format=node_dict["output_format"],
        device='cpu' if use_cpu else 'auto',
        use_tta=use_tta,
        store_dirs=node_dict["output_path"],
        logger=logger
    )    
    
    separator.process_folder(node_dict["input_path"])
    separator.del_cache()
    
    separator = None

# def main():
#     node_dict = {
#         'uid': 'bfc2ac81-a136-4616-b32d-419f125d6724', 
#         'model_name': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt', 
#         'model_type': 'bs_roformer', 
#         'path': './pretrain/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt', 
#         'config_path': 'configs/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.yaml', 
#         'input': ['input'], 'output': ['vocals', 'instrumental'], 
#         'parameter': [{'parameter': 'batch_size', 'type': 'int', 'default_value': 1, 'max_value': 100, 'min_value': 1, 'current_value': 1}, 
#                       {'parameter': 'dim_t', 'type': 'int', 'default_value': 901, 'max_value': 10000, 'min_value': 1, 'current_value': 901}, 
#                       {'parameter': 'num_overlap', 'type': 'int', 'default_value': 4, 'max_value': 100, 'min_value': 1, 'current_value': 4}], 
#         'bool': [{'parameter': 'use_cpu', 'default_value': False, 'current_value': False}, 
#                  {'parameter': 'use_tta', 'default_value': False, 'current_value': False}], 
#         'down_stream_nodes': [['54dc6a25-3848-440b-9382-05d3244ba56b', 1], ['2ec589b0-c3d5-4200-a114-08c0affde894', 0]],
#         'up_stream_node': 'ddf58dd7-a88e-405e-9a73-4ff73b8eccf8', 
#         'output_format': 'wav', 
#         'scene_pos': [-676.0, -405.0], 
#         'input_path': 'input/', 
#         'output_path': {'instrumental': 'output/instrument', 'vocals': './tmp\\bfc2ac81-a136-4616-b32d-419f125d6724_vocals'}}    
    
#     msst_inference(node_dict)

# if __name__ == '__main__':
#     main()    