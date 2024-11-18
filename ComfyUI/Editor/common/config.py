from qfluentwidgets import QConfig, ConfigItem, qconfig, OptionsConfigItem, OptionsValidator
from ComfyUI.Editor.common.language import Language, LanguageSerializer

class EditorConfig(QConfig):
    
    # aria2_port = ConfigItem("aria2", "Aria2_RPC_URL", 16800, restart=True)
    # aria2_secret = ConfigItem("aria2", "Aria2_RPC_SECRET", "", restart=True)
    # hf_endpoint = ConfigItem("huggingface", "HF_ENDPOINT", "https://hf-mirror.com", restart=True)
    language = OptionsConfigItem(
        "MainWindow", "Language", Language.AUTO, OptionsValidator(Language), LanguageSerializer(), restart=True
        )
    pan_sensitivity = ConfigItem("Editor", "PanSensitivity", 1.0, restart=True)
    
cfg = EditorConfig()
qconfig.load("./ComfyUI/Editor/data/EditorConfig.json", cfg)