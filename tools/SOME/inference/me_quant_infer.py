from typing import Dict, List

import numpy as np
import torch

from utils.infer_utils import decode_bounds_to_alignment, decode_note_sequence
from .me_infer import MIDIExtractionInference


class QuantizedMIDIExtractionInference(MIDIExtractionInference):
    @torch.no_grad()
    def forward_model(self, sample: Dict[str, torch.Tensor]):
        probs, bounds = self.model(x=sample['units'], f0=sample['pitch'], mask=sample['masks'], softmax=True)

        return {
            'probs': probs,
            'bounds': bounds,
            'masks': sample['masks'],
        }

    def postprocess(self, results: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
        probs = results['probs']
        bounds = results['bounds']
        masks = results['masks']
        probs *= masks[..., None]
        bounds *= masks
        unit2note_pred = decode_bounds_to_alignment(bounds) * masks
        midi_pred = probs.argmax(dim=-1)
        rest_pred = midi_pred == 128
        note_midi_pred, note_dur_pred, note_mask_pred = decode_note_sequence(
            unit2note_pred, midi_pred.clip(min=0, max=127), ~rest_pred & masks
        )
        note_rest_pred = ~note_mask_pred
        return {
            'note_midi': note_midi_pred.squeeze(0).cpu().numpy(),
            'note_dur': note_dur_pred.squeeze(0).cpu().numpy() * self.timestep,
            'note_rest': note_rest_pred.squeeze(0).cpu().numpy()
        }
