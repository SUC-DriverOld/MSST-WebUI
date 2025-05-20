from .base_infer import BaseInference
from .me_infer import MIDIExtractionInference
from .me_quant_infer import QuantizedMIDIExtractionInference

task_inference_mapping = {"training.MIDIExtractionTask": "MIDIExtractionInference", "training.QuantizedMIDIExtractionTask": "QuantizedMIDIExtractionInference"}
