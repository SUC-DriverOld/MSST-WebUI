import importlib
import pathlib

import click
import librosa
import yaml
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tools.SOME import inference
from tools.SOME.utils.infer_utils import build_midi_file
from tools.SOME.utils.slicer2 import Slicer

@click.command(help='Run inference with a trained model')
@click.option('--model', required=True, metavar='CKPT_PATH', help='Path to the model checkpoint (*.ckpt)')
@click.option('--wav', required=True, metavar='WAV_PATH', help='Path to the input wav file (*.wav)')
@click.option('--midi', required=False, metavar='MIDI_PATH', help='Path to the output MIDI file (*.mid)')
@click.option('--tempo', required=False, type=float, default=120, metavar='TEMPO', help='Specify tempo in the output MIDI')
def infer(model, wav, midi, tempo):
    model_path = pathlib.Path(model)
    with open(model_path.with_name('config.yaml'), 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    infer_cls = inference.task_inference_mapping[config['task_cls']]

    pkg = ".".join(infer_cls.split(".")[:-1])
    cls_name = infer_cls.split(".")[-1]
    infer_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(infer_cls, inference.BaseInference), \
        f'Inference class {infer_cls} is not a subclass of {inference.BaseInference}.'
    infer_ins = infer_cls(config=config, model_path=model_path)

    wav_path = pathlib.Path(wav)
    waveform, _ = librosa.load(wav_path, sr=config['audio_sample_rate'], mono=True)
    slicer = Slicer(sr=config['audio_sample_rate'], max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    midis = infer_ins.infer([c['waveform'] for c in chunks])

    midi_file = build_midi_file([c['offset'] for c in chunks], midis, tempo=tempo)

    midi_path = pathlib.Path(midi) if midi is not None else wav_path.with_suffix('.mid')
    midi_file.save(midi_path)


if __name__ == '__main__':
    infer()
