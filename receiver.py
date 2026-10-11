"""Conventional iterative MMSE-PIC/BCJR receiver."""

import numpy as np

from bcjr_codec import ConvolutionalCodec
from mmse_pic import mmse_pic
from tx_pipeline import SystemConfig, grid_to_stream, stream_to_grid


class ConventionalReceiver:
    def __init__(self, config=SystemConfig()):
        self.config = config
        self.codec = ConvolutionalCodec()

    def decode(self, y, h, n0, permutation):
        config = self.config
        feedback = np.zeros((1, config.n_tones, config.n_tx, config.bits_per_symbol))
        for _ in range(config.iterations):
            detector_llr = mmse_pic(y, h, n0, feedback)
            coded_app, info_bits = self.codec.decode(
                grid_to_stream(detector_llr, config), permutation)
            # The next detector round uses coded APP, including channel evidence.
            feedback = stream_to_grid(coded_app, config)
        return np.packbits(info_bits[:, :config.source_bits_per_block], bitorder="big").tobytes()
