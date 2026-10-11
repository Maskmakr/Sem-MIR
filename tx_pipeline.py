"""128-byte packet dimensions and the original antenna/bit mapping."""
from dataclasses import dataclass
import numpy as np
from mmse_pic import make_constellation

@dataclass(frozen=True)
class SystemConfig:
    n_tx: int = 4
    n_rx: int = 4
    iterations: int = 6

    packet_bytes = 128
    source_bits_per_block = 1024
    coded_bits = 1248
    bits_per_symbol = 4

    @property
    def n_tones(self):
        return 312 // self.n_tx

def stream_to_grid(stream, config=SystemConfig()):
    x = np.asarray(stream)
    # Convert antenna-major bit streams to the tone-major detector grid.
    return x.reshape(x.shape[:-1]+(config.n_tx,config.n_tones,config.bits_per_symbol)).swapaxes(-3,-2)

def grid_to_stream(grid, config=SystemConfig()):
    x = np.asarray(grid)
    return x.swapaxes(-3,-2).reshape(x.shape[:-3]+(config.coded_bits,))

def modulate(coded_bits, permutation, config=SystemConfig()):
    grid = stream_to_grid(np.asarray(coded_bits)[...,permutation],config)
    # QAM labels use least-significant-bit ordering within each symbol.
    index = np.sum(grid*(1<<np.arange(config.bits_per_symbol)),axis=-1)
    constellation,_ = make_constellation(config.n_tx)
    return constellation[index]
