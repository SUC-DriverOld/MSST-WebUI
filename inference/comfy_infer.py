from inference.msst_infer import MSSeparator
from modules.vocal_remover.separator import Separator

class ComfyMSST(MSSeparator):
    """ComfyMSST is a class for separating music sources using the MSST model.
    Currently, it is a subclass of MusicSourceSeparator, with no additional methods or attributes.
    """
    pass

class ComfyVR(Separator):
    """ComfyVR is a class for removing vocals from music using the VR model.
    Currently, it is a subclass of VocalRemover, with no additional methods or attributes.
    """
    pass