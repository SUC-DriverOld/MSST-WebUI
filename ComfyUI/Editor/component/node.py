"""
example of a node_dict:
{   
    "index": 0, # index of the node, default is -1
    "model_name": "model_bs_roformer_ep_317_sdr_12.9755.ckpt", # name of the model
    "model_type": "bs_roformer", # type of the model
    "path": "./pretrain/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "config_path: "./configs/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.yaml"
    "input: ["input"], # list of input ports
    "output": ["vocal", "instruments"], # list of output ports
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
            "default_value": 801,
            "max_value": 10000,
            "min_value": 1,
            "current_value": 801
        }
    ],
    "bool": [
        {
            "parameter": "Use TTA",
            "default_value": False,
            "current_value": False
        }
    ],
    down_stream_nodes: [[1, 0], [2, 1]], # list of downstream nodes, each element is a list of [index, output_port_index], default is []
    output_format: "wav", # output format of the node, in ["wav", "mp3", "flac"]
    scene_pos: [0, 0], # position of the node in the scene, default is [0, 0]
}
"""