import librosa
import yaml
import os

from tools.SOME import inference
from tools.SOME.utils.infer_utils import build_midi_file
from tools.SOME.utils.slicer2 import Slicer


def infer(model_path, config_path, wav_path, output_dir, tempo):
	with open(config_path, "r", encoding="utf8") as f:
		config = yaml.safe_load(f)
	infer_cls = inference.task_inference_mapping[config["task_cls"]]

	if infer_cls == "MIDIExtractionInference":
		from tools.SOME.inference.me_infer import MIDIExtractionInference

		infer_ins = MIDIExtractionInference(config=config, model_path=model_path)
	elif infer_cls == "QuantizedMIDIExtractionInference":
		from tools.SOME.inference.me_quant_infer import QuantizedMIDIExtractionInference

		infer_ins = QuantizedMIDIExtractionInference(config=config, model_path=model_path)
	else:
		raise ValueError(f"Unknown inference class: {infer_cls}")

	waveform, _ = librosa.load(wav_path, sr=config["audio_sample_rate"], mono=True)
	slicer = Slicer(sr=config["audio_sample_rate"], max_sil_kept=1000)
	chunks = slicer.slice(waveform)
	midis = infer_ins.infer([c["waveform"] for c in chunks])
	midi_file = build_midi_file([c["offset"] for c in chunks], midis, tempo=tempo)

	os.makedirs(output_dir, exist_ok=True)
	wav_name = os.path.splitext(os.path.basename(wav_path))[0]
	midi_path = os.path.join(output_dir, f"{wav_name}.mid")
	midi_file.save(midi_path)

	return midi_path
