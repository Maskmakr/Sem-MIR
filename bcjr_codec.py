"""Fixed 128-byte convolutional codec; positive LLR means bit one."""

import numpy as np
from numba import njit


# Six memory bits plus the current input feed the (133,171) generators.
NEXT_STATE = np.empty((64, 2), dtype=np.int64)
OUTPUT_SYMBOL = np.empty((64, 2), dtype=np.int64)
for _state in range(64):
    for _bit in range(2):
        _register = (_bit << 6) | _state
        NEXT_STATE[_state, _bit] = (_state >> 1) | (_bit << 5)
        OUTPUT_SYMBOL[_state, _bit] = (((_register & 0o133).bit_count() % 2) << 1) | ((_register & 0o171).bit_count() % 2)

# Rate 5/6 retains six outputs for each five input bits.
PUNCTURE = (np.arange(208)[:, None] * 10 + np.array([0, 1, 2, 5, 6, 9])).ravel()


@njit
def _encode(bits):
    out = np.empty(2080, dtype=np.uint8)
    state = 0
    for t in range(1040):
        bit = bits[t]
        symbol = OUTPUT_SYMBOL[state, bit]
        out[2*t] = symbol >> 1
        out[2*t+1] = symbol & 1
        state = NEXT_STATE[state, bit]
    return out


@njit
def _bcjr_one(llrs):
    branch = np.empty((1040, 4), dtype=np.float64)
    for t in range(1040):
        for q in range(4):
            branch[t, q] = 0.5 * ((1 if q & 2 else -1) * llrs[2*t] + (1 if q & 1 else -1) * llrs[2*t+1])

    # Prefix/suffix maxima retain the original finite boundary convention.
    alpha = np.full((1041, 64), -np.inf)
    alpha[0, 0] = 100000.0
    for t in range(1040):
        for state in range(64):
            for bit in range(2):
                # Ten filler bits and six termination bits are known zeros.
                if t >= 1024 and bit == 1:
                    continue
                nxt = NEXT_STATE[state, bit]
                metric = alpha[t, state] + branch[t, OUTPUT_SYMBOL[state, bit]]
                if metric > alpha[t+1, nxt]:
                    alpha[t+1, nxt] = metric

    beta = np.full(64, -np.inf)
    beta[0] = 100000.0
    coded = np.empty(2080, dtype=np.float64)
    bits = np.empty(1040, dtype=np.uint8)
    for t in range(1039, -1, -1):
        new_beta = np.full(64, -np.inf)
        imax = np.zeros(2, dtype=np.float64)
        cmax = np.zeros((2, 2), dtype=np.float64)
        for state in range(64):
            for bit in range(2):
                if t >= 1024 and bit == 1:
                    continue
                symbol = OUTPUT_SYMBOL[state, bit]
                bm = beta[NEXT_STATE[state, bit]] + branch[t, symbol]
                if bm > new_beta[state]:
                    new_beta[state] = bm
                metric = bm + alpha[t, state]
                if metric > imax[bit]:
                    imax[bit] = metric
                for j in range(2):
                    cb = (symbol >> (1-j)) & 1
                    if metric > cmax[j, cb]:
                        cmax[j, cb] = metric
        beta = new_beta
        bits[t] = (imax[1] - imax[0]) > 0
        coded[2*t] = cmax[0, 1] - cmax[0, 0]
        coded[2*t+1] = cmax[1, 1] - cmax[1, 0]
    return coded, bits


class ConvolutionalCodec:
    coded_bits = 1248
    trellis_bits = 1040

    def encode(self, information):
        bits = np.asarray(information, dtype=np.uint8).reshape(1040)
        return _encode(bits)[PUNCTURE][None, :]

    def decode(self, detector_llr, permutation):
        # Undo the interleaver; punctured positions have neutral channel LLRs.
        deinterleaved = np.empty(1248, dtype=np.float64)
        deinterleaved[permutation] = np.asarray(detector_llr).reshape(1248)
        mother = np.zeros(2080)
        mother[PUNCTURE] = deinterleaved
        coded, bits = _bcjr_one(mother)
        # PIC consumes full coded APPs in transmitted order.
        return coded[PUNCTURE][permutation][None, :], bits[None, :]
