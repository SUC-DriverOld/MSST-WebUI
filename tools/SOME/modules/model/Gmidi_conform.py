import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.conform.Gconform import Gmidi_conform



class midi_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, x, target):
        midiout, cutp = x
        midi_target, cutp_target = target

        cutploss = self.loss(cutp, cutp_target)
        midiloss = self.loss(midiout, midi_target)
        return midiloss, cutploss


class midi_conforms(nn.Module):
    def __init__(self, config):
        super().__init__()

        cfg = config['midi_extractor_args']
        cfg.update({'indim': config['units_dim'], 'outdim': config['midi_num_bins']})
        self.model = Gmidi_conform(**cfg)

    def forward(self, x, f0, mask=None,softmax=False,sig=False):

        midi,bound=self.model(x, f0, mask)
        if  sig:
            midi = torch.sigmoid(midi)

        if softmax:
            midi=F.softmax(midi,dim=2)


        return midi,bound

    def get_loss(self):
        return midi_loss()
