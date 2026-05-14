"""
transmitter.py
--------------
Complete satellite downlink transmitter chain:

  1. Generate random bits
  2. Map bits → 16-QAM symbols
  3. Square-root raised cosine Tx filter  (upsample + shape)
  4. DPD pre-distortion                  (optional – new)
  5. Saleh TWTA HPA                      (optional – can bypass with apply_hpa=False)

Changes vs original Python
---------------------------
* DPD (Digital Pre-Distortion) is now a real implemented block using
  saleh_dpd() from impairments.py.  Corresponds to the DPD subsystem in
  the MATLAB RF Satellite Link model.
* apply_hpa=False bypasses the HPA entirely (ideal linear amplifier).
  MATLAB equivalent: disconnecting the TWTA block.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import Config
from modulation import bits_to_symbols
from filters import srrc_coeffs, tx_filter
from impairments import saleh_hpa, saleh_dpd


@dataclass
class TxSignals:
    """Container for all transmitter stage outputs."""
    bits:         np.ndarray    # raw bit stream
    symbols:      np.ndarray    # complex symbols (1/sym)
    filtered:     np.ndarray    # after Tx SRRC filter (oversampled)
    after_dpd:    np.ndarray    # after DPD (= filtered if DPD off)
    after_hpa:    np.ndarray    # after Saleh HPA (or bypass)
    srrc_h:       np.ndarray    # filter coefficients for Rx reuse
    dpd_applied:  bool = False  # diagnostic flag


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
    TxSignals dataclass with all intermediate waveforms.
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

    # Normalise to unit average power before HPA / DPD
    rms = float(np.sqrt(np.mean(np.abs(filtered) ** 2)))
    filtered_norm = filtered / rms if rms > 0 else filtered.copy()

    ibo = custom_backoff_db if custom_backoff_db is not None else cfg.hpa_input_backoff_db

    # ------------------------------------------------------------------
    # 4. Digital Pre-Distortion  (DPD)  – MATLAB DPD subsystem
    # ------------------------------------------------------------------
    dpd_applied = False
    if cfg.apply_hpa and cfg.apply_dpd:
        after_dpd = saleh_dpd(
            filtered_norm,
            a_a=cfg.hpa_saleh_a_a,
            b_a=cfg.hpa_saleh_b_a,
            a_p=cfg.hpa_saleh_a_p,
            b_p=cfg.hpa_saleh_b_p,
            input_backoff_db=ibo,
            lut_points=cfg.dpd_lut_points,
        )
        dpd_applied = True
        if cfg.verbose:
            print(f"  [Tx] DPD applied (IBO={ibo:.1f} dB, LUT={cfg.dpd_lut_points} pts)")
    else:
        after_dpd = filtered_norm

    # ------------------------------------------------------------------
    # 5. HPA – Saleh TWTA  (or bypass)
    # ------------------------------------------------------------------
    if cfg.apply_hpa:
        after_hpa = saleh_hpa(
            after_dpd,
            a_a=cfg.hpa_saleh_a_a,
            b_a=cfg.hpa_saleh_b_a,
            a_p=cfg.hpa_saleh_a_p,
            b_p=cfg.hpa_saleh_b_p,
            input_backoff_db=ibo,
        )
        if cfg.verbose:
            tx_pwr_db = 10 * np.log10(np.mean(np.abs(after_hpa) ** 2) + 1e-30)
            print(f"  [Tx] HPA ON  IBO={ibo:.1f} dB | Tx power: {tx_pwr_db:.2f} dB (norm)")
    else:
        # Bypass: scale to the same average power as a linear HPA would produce
        after_hpa = after_dpd.copy()
        if cfg.verbose:
            print(f"  [Tx] HPA BYPASS (ideal linear amplifier)")

    return TxSignals(
        bits=bits,
        symbols=symbols,
        filtered=filtered_norm,
        after_dpd=after_dpd,
        after_hpa=after_hpa,
        srrc_h=h,
        dpd_applied=dpd_applied,
    )
