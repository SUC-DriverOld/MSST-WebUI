import json
import os
import yaml
from ml_collections import ConfigDict
from omegaconf import OmegaConf

model_info_path = "./data/models_info.json"

def generate_cfg():
    with open(model_info_path, "r") as f:
        model_info = json.load(f)

    for model in model_info:
        row = model_info[model]
        model_name = row["model_name"]
        model_class = row["model_class"]
        target_position = row["target_position"]
        
        model_dict = {
            "uid": None,
            "model_name": model_name,
            "input": {
                "input": {
                    "connection": "source_node_id.output_port_id",
                    "required": True
                },
            },
            "output": {
                
            },
        }
        
        if model_class == "VR_Models":
            primary_stem = row["primary_stem"]
            secondary_stem = row["secondary_stem"]
            model_dict["model_type"] = None
            model_dict["path"] = target_position
            model_dict["primary_stem"] = primary_stem
            model_dict["secondary_stem"] = secondary_stem
            model_dict["output"] = {
                primary_stem: {
                    "connection": None,
                    "required": False
                },
                secondary_stem: {
                    "connection": None,
                    "required": False
                }
            }
            model_dict["parameter"] = [
                {
                    "parameter": "batch_size",
                    "type": "int",
                    "default_value": 2,
                    "max_value": 100,
                    "min_value": 1,
                    "current_value": 2
                },
                {
                    "parameter": "window_size",
                    "type": "int",
                    "default_value": 512,
                    "max_value": 10000,
                    "min_value": 1,
                    "current_value": 512
                },
                {
                    "parameter": "aggression",
                    "type": "int",
                    "default_value": 5,
                    "max_value": 100,
                    "min_value": -100,
                    "current_value": 5
                },
                {
                    "parameter": "post_process_threshold",
                    "type": "float",
                    "default_value": 0.2,
                    "max_value": 0.3,
                    "min_value": 0.1,
                    "current_value": 0.2
                }
            ]
            model_dict["bool"] = [
                {
                    "parameter": "use_cpu",
                    "default_value": False,
                    "current_value": False
                },
                {
                    "parameter": "invert_spect",
                    "default_value": False,
                    "current_value": False
                },
                {
                    "parameter": "enable_tta",
                    "default_value": False,
                    "current_value": False
                },
                {
                    "parameter": "high_end_process",
                    "default_value": False,
                    "current_value": False
                },
                {
                    "parameter": "enable_post_process",
                    "default_value": False,
                    "current_value": False
                }
            ]

        else:
            model_type = row["model_type"]
            model_dict["model_type"] = model_type
            model_dict["path"] = target_position
            config_path = target_position.replace('pretrain', 'configs') + '.yaml'
            with open(config_path) as f:
                if model_type == 'htdemucs':
                    config = OmegaConf.load(config_path)
                else:
                    config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            for instrument in config.training.instruments:
                model_dict["output"][instrument] = {
                    "connection": None,
                    "required": False
                }
            model_dict["parameter"] = []
            if "batch_size" in config.inference:
                model_dict["parameter"].append({
                    "parameter": "batch_size",
                    "type": "int",
                    "default_value": config.inference.batch_size,
                    "max_value": 100,
                    "min_value": 1,
                    "current_value": config.inference.batch_size
                })
            if "dim_t" in config.inference:
                model_dict["parameter"].append({
                    "parameter": "dim_t",
                    "type": "int",
                    "default_value": config.inference.dim_t,
                    "max_value": 10000,
                    "min_value": 1,
                    "current_value": config.inference.dim_t
                })
            if "num_overlap" in config.inference:
                model_dict["parameter"].append({
                    "parameter": "num_overlap",
                    "type": "int",
                    "default_value": config.inference.num_overlap,
                    "max_value": 100,
                    "min_value": 1,
                    "current_value": config.inference.num_overlap
                })

            model_dict["bool"] = [
                {
                    "parameter": "use_cpu",
                    "default_value": False,
                    "current_value": False
                },
                {
                    "parameter": "use_tta",
                    "default_value": False,
                    "current_value": False
                }
            ]
            if "normalize" in config.inference:
                model_dict["bool"].append({
                    "parameter": "normalize",
                    "default_value": False,
                    "current_value": False
                })
            break

        model_dict["output_format"] = "wav"
        model_dict["scene_pos"] = [0, 0]
        model_dict["input_path"] = None
        model_dict["output_path"] = None

        # print(model_dict)
        try:
            with open(f"./ComfyUI/Editor/data/nodes/{model_name}.json", "w") as f:
                json.dump(model_dict, f, indent=4)
        except Exception as e:
            print(e)
            print(model_dict)

if __name__ == "__main__":
    generate_cfg()