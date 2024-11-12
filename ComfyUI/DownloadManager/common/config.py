from qfluentwidgets import QConfig, ConfigItem, qconfig


class AppConfig(QConfig):
    
    aria2_port = ConfigItem("aria2", "Aria2_RPC_URL", 16800, restart=True)
    hf_endpoint = ConfigItem("huggingface", "HF_ENDPOINT", "https://hf-mirror.com")
    
cfg = AppConfig()
qconfig.load("./data/AppConfig.json", cfg)