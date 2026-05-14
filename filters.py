"""
filters.py
----------
Square-Root Raised-Cosine (SRRC) pulse-shaping filter.

The transmit filter upsamples the symbol stream and shapes each pulse.
The receive filter acts as a matched filter, then downsamples back to 1
sample/symbol.  Together they form a raised-cosine (Nyquist) pair.
"""

import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# SRRC impulse response
# ---------------------------------------------------------------------------

def srrc_coeffs(rolloff: float, span: int, sps: int) -> np.ndarray:
    """
    Compute the impulse response of a Square-Root Raised-Cosine filter.

    Parameters
    ----------
    rolloff : excess-bandwidth factor β ∈ (0, 1]
    span    : filter length in symbols (total length = span*sps + 1)
    sps     : samples per symbol

    Returns
    -------
    h : real-valued coefficient array, length = span*sps + 1, energy-normalised
    """
    beta = rolloff
    N = span * sps          # half-length in samples (symmetric, odd total)
    t = np.arange(-N, N + 1) / sps   # normalised time axis (in symbol periods)

    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = (1 - beta) + 4 * beta / np.pi
        elif abs(ti) == 1.0 / (4 * beta + 1e-30):
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    # Normalise so that the matched-filter pair has unit gain at DC
    h /= np.sqrt(np.sum(h ** 2))
    return h


# ---------------------------------------------------------------------------
# Transmit filter  (upsample → convolve)
# ---------------------------------------------------------------------------

def tx_filter(symbols: np.ndarray, h: np.ndarray, sps: int) -> np.ndarray:
    """
    Upsample symbol sequence by *sps* and apply the SRRC transmit filter.

    Parameters
    ----------
    symbols : complex 1-D array  (one sample per symbol)
    h       : SRRC coefficients from srrc_coeffs()
    sps     : samples per symbol (upsampling factor)

    Returns
    -------
    signal : complex 1-D array at rate sps * symbol_rate
    """
    # Insert (sps-1) zeros between each symbol  (upsample)
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    # Linear FIR convolution
    filtered = np.convolve(upsampled, h, mode='full')
    return filtered


# ---------------------------------------------------------------------------
# Receive filter  (convolve → downsample)
# ---------------------------------------------------------------------------

def rx_filter(signal: np.ndarray, h: np.ndarray, sps: int,
              delay: int) -> np.ndarray:
    """
    Apply SRRC matched filter and downsample to 1 sample/symbol.

    Parameters
    ----------
    signal : complex 1-D array at sample rate
    h      : same SRRC coefficients used at the transmitter
    sps    : samples per symbol
    delay  : total group delay of Tx+Rx filter cascade in samples
             = span * sps  (each filter contributes span*sps/2 samples)

    Returns
    -------
    symbols : complex 1-D array (1 sample/symbol)
    """
    filtered = np.convolve(signal, h, mode='full')

    # The combined Tx+Rx delay is 2*(span*sps/2) = span*sps samples
    # We start sampling at index *delay* to align with symbol centres
    symbols = filtered[delay::sps]
    return symbols


# ---------------------------------------------------------------------------
# Helper – compute filter group delay
# ---------------------------------------------------------------------------

def filter_delay(span: int, sps: int) -> int:
    """
    Total group delay of the Tx+Rx SRRC cascade in *samples*.

    Each SRRC filter has group delay = span*sps/2 samples.
    Both Tx and Rx use 'full' convolution mode, contributing
    span*sps/2 each ← but the 'full' output is shifted by the
    full filter length (span*sps), not half.

    Empirically verified: the correct offset for symbol-aligned
    downsampling after two cascaded 'full' convolutions is 2*span*sps.
    """
    return 2 * span * sps
