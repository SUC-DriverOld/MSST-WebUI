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

from inference.comfy_infer import ComfyMSST
from ml_collections import ConfigDict
from omegaconf import OmegaConf
import yaml

def msst_inference(node_dict):
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
        yaml.dump(config.to_dict(), f)    
        
    separator = ComfyMSST(
        model_type=model_type,
        model_path=node_dict["path"],
        config_path=config_path,
        output_format=node_dict["output_format"],
        device='cpu' if use_cpu else 'auto',
        use_tta=use_tta,
        store_dirs=node_dict["output_path"]
    )    
    
    separator.process_folder(node_dict["input_path"])
    separator.del_cache()
    
    separator = None