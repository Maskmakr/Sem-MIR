"""Transmit one text packet and return the receiver's bytes."""

from pathlib import Path

import numpy as np

from receiver import ConventionalReceiver
from tx_pipeline import SystemConfig, modulate


def run(payload, ebn0_db=7.0, seed=0, config=SystemConfig()):
    receiver = ConventionalReceiver(config)
    permutation = np.random.default_rng(
        np.random.SeedSequence([128260912, 0])).permutation(config.coded_bits)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))

    # The payload is followed by ten zero filler bits and six terminating zeros.
    source = np.zeros((1, receiver.codec.trellis_bits), dtype=np.uint8)
    source[:, :config.source_bits_per_block] = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    h = (rng.standard_normal((config.n_rx, config.n_tx))
         + 1j * rng.standard_normal((config.n_rx, config.n_tx))) / np.sqrt(2)
    symbols = modulate(receiver.codec.encode(source), permutation, config)
    clean = np.einsum("rt,bst->bsr", h, symbols)
    noise = (rng.standard_normal(clean.shape)
             + 1j * rng.standard_normal(clean.shape)) / np.sqrt(2)

    # Unit transmit-vector energy includes the cost of filler and termination.
    offset_db = float(10 * np.log10(config.source_bits_per_block / config.n_tones))
    n0 = float(np.power(10.0, -(ebn0_db + offset_db) / 10))
    return receiver.decode(clean + np.sqrt(n0) * noise, h, n0, permutation)


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data" / "paper128_test.txt"
    payload = path.read_text(encoding="ascii").encode("ascii")[:128]
    print(run(payload))
