"""
modulation.py
-------------
16-QAM Gray-coded symbol mapper and hard-decision demapper.

Constellation layout follows the standard Gray-coded square QAM grid used
in MATLAB's comm.RectangularQAMModulator (unit average power, normalised).
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Build the Gray-coded 16-QAM constellation lookup tables once at import time
# ---------------------------------------------------------------------------

def _build_qam16_tables():
    """
    Return (symbols, bit_table) where
      symbols    : complex array, length 16, average power ≈ 1.0
      bit_table  : int array shape (16, 4), each row is the 4-bit Gray word
    """
    # 1-D Gray-coded PAM-4 levels: 00→-3, 01→-1, 11→+1, 10→+3  (Gray order)
    # Mapping: index 0..3 → amplitude level in Gray code order
    gray_map_1d = np.array([-3, -1, 1, 3], dtype=float)  # amplitudes
    # Gray-code words for indices 0..3:  00, 01, 11, 10
    gray_bits_1d = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=int)

    symbols = np.zeros(16, dtype=complex)
    bits = np.zeros((16, 4), dtype=int)

    idx = 0
    for i in range(4):          # I (in-phase) axis
        for q in range(4):      # Q (quadrature) axis
            symbols[idx] = gray_map_1d[i] + 1j * gray_map_1d[q]
            bits[idx, :2] = gray_bits_1d[i]
            bits[idx, 2:] = gray_bits_1d[q]
            idx += 1

    # Normalise to unit average power
    avg_power = np.mean(np.abs(symbols) ** 2)
    symbols /= np.sqrt(avg_power)

    return symbols, bits


SYMBOLS_16QAM, BITS_16QAM = _build_qam16_tables()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bits_to_symbols(bits: np.ndarray, order: int = 16) -> np.ndarray:
    """
    Map a 1-D bit stream to complex QAM symbols.

    Parameters
    ----------
    bits  : 1-D int array, length must be a multiple of log2(order)
    order : QAM order (only 16 supported here)

    Returns
    -------
    symbols : 1-D complex array, length = len(bits) / log2(order)
    """
    if order != 16:
        raise NotImplementedError("Only 16-QAM is implemented.")

    bits = np.asarray(bits, dtype=int).ravel()
    bps = 4
    n_sym = len(bits) // bps
    bits = bits[: n_sym * bps].reshape(n_sym, bps)

    # Build a (16,4) lookup for fast index search
    # Convert each row of BITS_16QAM to an integer key
    keys = np.packbits(BITS_16QAM, axis=1, bitorder='big')[:, 0] >> 4  # 4-bit values
    input_keys = np.packbits(bits, axis=1, bitorder='big')[:, 0] >> 4

    # Build reverse map: 4-bit integer → symbol
    lut = np.zeros(16, dtype=complex)
    for k, sym in zip(keys, SYMBOLS_16QAM):
        lut[k] = sym

    return lut[input_keys]


def symbols_to_bits(symbols: np.ndarray, order: int = 16) -> np.ndarray:
    """
    Hard-decision 16-QAM demodulation.

    Parameters
    ----------
    symbols : 1-D complex array of received (possibly noisy) symbols
    order   : QAM order

    Returns
    -------
    bits : 1-D int array, length = 4 * len(symbols)
    """
    if order != 16:
        raise NotImplementedError("Only 16-QAM is implemented.")

    symbols = np.asarray(symbols).ravel()
    n_sym = len(symbols)

    # Nearest-neighbour decision via Euclidean distance
    # Shape broadcast: (n_sym, 1) vs (1, 16)
    dist = np.abs(symbols[:, np.newaxis] - SYMBOLS_16QAM[np.newaxis, :]) ** 2
    idx = np.argmin(dist, axis=1)  # (n_sym,)

    bits = BITS_16QAM[idx].ravel()  # (4*n_sym,)
    return bits.astype(int)


def symbol_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray,
                      bits_per_symbol: int = 4) -> Tuple[float, int, int]:
    """
    Compute BER and SER from aligned bit streams.

    Returns
    -------
    ber, ser, num_bit_errors
    """
    n = min(len(tx_bits), len(rx_bits))
    # Align to integer number of symbols
    n = (n // bits_per_symbol) * bits_per_symbol
    tx = tx_bits[:n]
    rx = rx_bits[:n]

    bit_errors = int(np.sum(tx != rx))
    ber = bit_errors / n if n > 0 else 0.0

    # SER: at least one bit wrong in a symbol
    tx_sym = tx.reshape(-1, bits_per_symbol)
    rx_sym = rx.reshape(-1, bits_per_symbol)
    sym_errors = int(np.sum(np.any(tx_sym != rx_sym, axis=1)))
    ser = sym_errors / (n // bits_per_symbol) if n > 0 else 0.0

    return ber, ser, bit_errors
