"""
{
    "uid": null,
    "model_name": "4_HP-Vocal-UVR.pth",
    "model_type": null,
    "path": "./pretrain/VR_Models/4_HP-Vocal-UVR.pth",
    "config_path": null,
    "input": [
        "input"
    ],
    "output": [
        "Vocals",
        "Instrumental"
    ],
    "parameter": [
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
    ],
    "bool": [
        {
            "parameter": "use_cpu",
            "default_value": false,
            "current_value": false
        },
        {
            "parameter": "invert_spect",
            "default_value": false,
            "current_value": false
        },
        {
            "parameter": "enable_tta",
            "default_value": false,
            "current_value": false
        },
        {
            "parameter": "high_end_process",
            "default_value": false,
            "current_value": false
        },
        {
            "parameter": "enable_post_process",
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


from inference.comfy_infer import ComfyVR

def vr_inference(node_dict, logger=None):
    model_path = node_dict["path"]
    input_path = node_dict["input_path"]
    output_dir = node_dict["output_path"]
    output_format = node_dict["output_format"]
    invert_using_spec = False
    use_cpu = False
    vr_params={
        "batch_size": 2, 
        "window_size": 512, 
        "aggression": 5, 
        "enable_tta": False, 
        "enable_post_process": False, 
        "post_process_threshold": 0.2, 
        "high_end_process": False
    }
    
    for param in node_dict["parameter"]:
        if param["parameter"] in vr_params:
            vr_params[param["parameter"]] = param["current_value"]
            
    for param in node_dict["bool"]:
        if param["parameter"] == "invert_spect":
            invert_using_spec = param["current_value"]
        elif param["parameter"] == "use_cpu":
            use_cpu = param["current_value"]
            
        elif param["parameter"] in vr_params:
            vr_params[param["parameter"]] = param["current_value"]
            
    separator = ComfyVR(
        model_file=model_path,
        output_dir=output_dir,
        output_format=output_format,
        invert_using_spec=invert_using_spec,
        use_cpu=use_cpu,
        vr_params=vr_params,
        logger=logger
    )            
    
    separator.process_folder(input_folder=input_path)
    separator.del_cache()
    separator = None