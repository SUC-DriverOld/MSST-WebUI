import pathlib
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import tqdm
from torch import nn

from tools.SOME.utils import build_object_from_class_name


class BaseInference:
	def __init__(self, config: dict, model_path: pathlib.Path, device=None):
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		self.config = config
		self.model_path = model_path
		self.device = device
		self.timestep = self.config["hop_size"] / self.config["audio_sample_rate"]
		self.model: torch.nn.Module = self.build_model()

	def build_model(self) -> nn.Module:
		model: nn.Module = build_object_from_class_name(self.config["model_cls"], nn.Module, config=self.config).eval().to(self.device)
		state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)["state_dict"]
		prefix_in_ckpt = "model"
		state_dict = OrderedDict({k[len(prefix_in_ckpt) + 1 :]: v for k, v in state_dict.items() if k.startswith(f"{prefix_in_ckpt}.")})
		model.load_state_dict(state_dict, strict=True)
		print(f"load '{prefix_in_ckpt}' from '{self.model_path}'.")
		return model

	def preprocess(self, waveform: np.ndarray) -> Dict[str, torch.Tensor]:
		raise NotImplementedError()

	def forward_model(self, sample: Dict[str, torch.Tensor]):
		raise NotImplementedError()

	def postprocess(self, results: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
		raise NotImplementedError()

	def infer(self, waveforms: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
		results = []
		for w in tqdm.tqdm(waveforms):
			model_in = self.preprocess(w)
			model_out = self.forward_model(model_in)
			res = self.postprocess(model_out)
			results.append(res)
		return results
