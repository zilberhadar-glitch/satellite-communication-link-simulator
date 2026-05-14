"""
transmitter.py
--------------
Complete satellite downlink transmitter chain:

  1. Generate random bits
  2. Map bits → 16-QAM symbols
  3. Square-root raised cosine Tx filter  (upsample + shape)
  4. Saleh TWTA HPA
  5. (Optional) Digital pre-distortion  [placeholder]
  6. Apply Tx antenna gain  (folded into path-loss module)

Returns intermediate signals at each stage so they can be plotted.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import Config
from modulation import bits_to_symbols
from filters import srrc_coeffs, tx_filter
from impairments import saleh_hpa


@dataclass
class TxSignals:
    """Container for all transmitter stage outputs."""
    bits: np.ndarray              # raw bit stream
    symbols: np.ndarray           # complex symbols (1/sym)
    filtered: np.ndarray          # after Tx SRRC filter  (oversampled)
    after_hpa: np.ndarray         # after Saleh HPA       (oversampled)
    srrc_h: np.ndarray            # filter coefficients  (for Rx reuse)


def transmit(cfg: Config,
             rng: np.random.Generator,
             custom_backoff_db: Optional[float] = None) -> TxSignals:
    """
    Run the full transmitter chain.

    Parameters
    ----------
    cfg               : simulation Config
    rng               : seeded numpy random generator
    custom_backoff_db : override cfg.hpa_input_backoff_db if given

    Returns
    -------
    TxSignals dataclass with all intermediate waveforms
    """
    # ------------------------------------------------------------------
    # 1. Generate random bits
    # ------------------------------------------------------------------
    n_bits = cfg.num_symbols * cfg.bits_per_symbol
    bits = rng.integers(0, 2, size=n_bits, dtype=int)

    # ------------------------------------------------------------------
    # 2. Map bits → 16-QAM symbols
    # ------------------------------------------------------------------
    symbols = bits_to_symbols(bits, order=cfg.modulation_order)

    # ------------------------------------------------------------------
    # 3. SRRC Tx filter (upsample + pulse shape)
    # ------------------------------------------------------------------
    h = srrc_coeffs(cfg.rolloff, cfg.span, cfg.samples_per_symbol)
    filtered = tx_filter(symbols, h, cfg.samples_per_symbol)

    # Normalise to unit average power before the HPA
    rms = np.sqrt(np.mean(np.abs(filtered) ** 2))
    if rms > 0:
        filtered_norm = filtered / rms
    else:
        filtered_norm = filtered.copy()

    # ------------------------------------------------------------------
    # 4. HPA – Saleh TWTA memoryless nonlinearity
    # ------------------------------------------------------------------
    ibo = custom_backoff_db if custom_backoff_db is not None else cfg.hpa_input_backoff_db
    after_hpa = saleh_hpa(
        filtered_norm,
        a_a=cfg.hpa_saleh_a_a,
        b_a=cfg.hpa_saleh_b_a,
        a_p=cfg.hpa_saleh_a_p,
        b_p=cfg.hpa_saleh_b_p,
        input_backoff_db=ibo,
    )

    if cfg.verbose:
        tx_pwr_dBm = 10 * np.log10(np.mean(np.abs(after_hpa) ** 2) + 1e-30)
        print(f"  [Tx] IBO={ibo:.1f} dB | Tx signal power: {tx_pwr_dBm:.2f} dB (normalised)")

    return TxSignals(
        bits=bits,
        symbols=symbols,
        filtered=filtered_norm,
        after_hpa=after_hpa,
        srrc_h=h,
    )
