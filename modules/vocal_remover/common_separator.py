""" This file contains the CommonSeparator class, common to all architecture-specific Separator classes. """

from logging import Logger
import gc
import numpy as np
import librosa
import torch

class CommonSeparator:
    """
    This class contains the common methods and attributes common to all architecture-specific Separator classes.
    """

    ALL_STEMS = "All Stems"
    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"
    NO_OTHER_STEM = "No Other"
    NO_BASS_STEM = "No Bass"
    NO_DRUM_STEM = "No Drums"
    NO_GUITAR_STEM = "No Guitar"
    NO_PIANO_STEM = "No Piano"
    NO_SYNTH_STEM = "No Synthesizer"
    NO_STRINGS_STEM = "No Strings"
    NO_WOODWINDS_STEM = "No Woodwinds"
    NO_WIND_INST_STEM = "No Wind Inst"
    NO_BRASS_STEM = "No Brass"
    PRIMARY_STEM = "Primary Stem"
    SECONDARY_STEM = "Secondary Stem"
    LEAD_VOCAL_STEM = "lead_only"
    BV_VOCAL_STEM = "backing_only"
    LEAD_VOCAL_STEM_I = "with_lead_vocals"
    BV_VOCAL_STEM_I = "with_backing_vocals"
    LEAD_VOCAL_STEM_LABEL = "Lead Vocals"
    BV_VOCAL_STEM_LABEL = "Backing Vocals"

    NON_ACCOM_STEMS = (VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM, GUITAR_STEM, PIANO_STEM, SYNTH_STEM, STRINGS_STEM, WOODWINDS_STEM, BRASS_STEM, WIND_INST_STEM)

    def __init__(self, config):

        self.logger: Logger = config.get("logger")
        self.debug = config.get("debug")

        # Inferencing device / acceleration config
        self.torch_device = config.get("torch_device")
        self.torch_device_cpu = config.get("torch_device_cpu")
        self.torch_device_mps = config.get("torch_device_mps")

        # Model data
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")

        # Output directory and format
        self.output_dir = config.get("output_dir")
        self.output_format = config.get("output_format")

        # Functional options which are applicable to all architectures and the user may tweak to affect the output
        self.invert_using_spec = config.get("invert_using_spec")
        self.sample_rate = config.get("sample_rate")

        # Model specific properties
        self.primary_stem_name = self.model_data.get("primary_stem", "primary_stem")
        self.secondary_stem_name = self.model_data.get("secondary_stem", "secondary_stem")

        self.is_karaoke = self.model_data.get("is_karaoke", False)
        self.is_bv_model = self.model_data.get("is_bv_model", False)
        self.bv_model_rebalance = self.model_data.get("is_bv_model_rebalanced", 0)

        self.logger.info(f"Common params: model_name={self.model_name}, model_path={self.model_path}")
        self.logger.info(f"Common params: output_dir={self.output_dir}, output_format={self.output_format}")
        self.logger.info(f"Common params: primary_stem_name={self.primary_stem_name}, secondary_stem_name={self.secondary_stem_name}")
        self.logger.debug(f"Common params: invert_using_spec={self.invert_using_spec}, sample_rate={self.sample_rate}")
        self.logger.debug(f"Common params: is_karaoke={self.is_karaoke}, is_bv_model={self.is_bv_model}, bv_model_rebalance={self.bv_model_rebalance}")

        # File-specific variables which need to be cleared between processing different audio inputs
        self.audio_file_path = None
        # self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None
        self.cached_sources_map = {}

    def separate(self, audio_file):
        """
        Placeholder method for separating audio sources. Should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def cached_sources_clear(self):
        """
        Clears the cache dictionaries for VR, MDX, and Demucs models.

        This function is essential for ensuring that the cache does not hold outdated or irrelevant data
        between different processing sessions or when a new batch of audio files is processed.
        It helps in managing memory efficiently and prevents potential errors due to stale data.
        """
        self.cached_sources_map = {}

    def cached_source_callback(self, model_architecture, model_name=None):
        """
        Retrieves the model and sources from the cache based on the processing method and model name.

        Args:
            model_architecture: The architecture type (VR, MDX, or Demucs) being used for processing.
            model_name: The specific model name within the architecture type, if applicable.

        Returns:
            A tuple containing the model and its sources if found in the cache; otherwise, None.

        This function is crucial for optimizing performance by avoiding redundant processing.
        If the requested model and its sources are already in the cache, they can be reused directly,
        saving time and computational resources.
        """
        model, sources = None, None

        mapper = self.cached_sources_map[model_architecture]

        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value

        return model, sources

    def cached_model_source_holder(self, model_architecture, sources, model_name=None):
        """
        Update the dictionary for the given model_architecture with the new model name and its sources.
        Use the model_architecture as a key to access the corresponding cache source mapper dictionary.
        """
        self.cached_sources_map[model_architecture] = {**self.cached_sources_map.get(model_architecture, {}), **{model_name: sources}}

    def prepare_mix(self, mix):
        """
        Prepares the mix for processing. This includes loading the audio from a file if necessary,
        ensuring the mix is in the correct format, and converting mono to stereo if needed.
        """
        # Store the original path or the mix itself for later checks
        audio_path = mix

        # Check if the input is a file path (string) and needs to be loaded
        if not isinstance(mix, np.ndarray):
            self.logger.debug(f"Loading audio from file: {mix}")
            mix, sr = librosa.load(mix, mono=False, sr=self.sample_rate)
            self.logger.debug(f"Audio loaded. Sample rate: {sr}, Audio shape: {mix.shape}")
        else:
            # Transpose the mix if it's already an ndarray (expected shape: [channels, samples])
            self.logger.debug("Transposing the provided mix array.")
            mix = mix.T
            self.logger.debug(f"Transposed mix shape: {mix.shape}")

        # If the original input was a filepath, check if the loaded mix is empty
        if isinstance(audio_path, str):
            if not np.any(mix):
                error_msg = f"Audio file {audio_path} is empty or not valid"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("Audio file is valid and contains data.")

        # Ensure the mix is in stereo format
        if mix.ndim == 1:
            self.logger.debug("Mix is mono. Converting to stereo.")
            mix = np.asfortranarray([mix, mix])
            self.logger.debug("Converted to stereo mix.")

        # Final log indicating successful preparation of the mix
        self.logger.debug("Mix preparation completed.")
        return mix

    def clear_file_specific_paths(self):
        """
        Clears the file-specific variables which need to be cleared between processing different audio inputs.
        """
        self.logger.debug("Clearing input audio file paths, sources and stems...")

        self.primary_source = None
        self.secondary_source = None
