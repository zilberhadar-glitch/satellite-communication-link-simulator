"""
receiver.py
-----------
Complete satellite downlink receiver chain:

  1. SRRC matched filter + downsample
  2. DC offset correction          (optional)
  3. AGC (Automatic Gain Control)  (optional)
  4. I/Q imbalance compensation    (optional)
  5. Doppler / carrier-frequency correction (optional)
  6. 16-QAM hard-decision demodulation
  7. BER / SER calculation

All compensators use simple closed-form estimates rather than adaptive
algorithms so the behaviour is deterministic and easy to understand.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from filters import rx_filter, filter_delay
from modulation import symbols_to_bits, symbol_error_rate


# ---------------------------------------------------------------------------
# Compensator helpers
# ---------------------------------------------------------------------------

def _dc_correction(symbols: np.ndarray) -> np.ndarray:
    """Remove estimated DC bias (sample mean)."""
    return symbols - np.mean(symbols)


def _agc(symbols: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """Scale symbols so that their average power equals *target_power*."""
    pwr = np.mean(np.abs(symbols) ** 2)
    if pwr > 0:
        return symbols * np.sqrt(target_power / pwr)
    return symbols


def _iq_imbalance_correction(symbols: np.ndarray) -> np.ndarray:
    """
    Blind I/Q imbalance compensation using the second-order statistics method.

    Given  y = I_rx + j Q_rx  with amplitude ε and phase Δφ imbalance:
        E[y²]  ≠  0   (would be zero for balanced I/Q)

    We estimate and remove the conjugate image component.

    Reference: Valkama et al., "Advanced methods for I/Q imbalance compensation
    in communication receivers", IEEE Trans. Signal Proc., 2001.
    """
    # Estimate second-order statistics
    c1 = np.mean(np.abs(symbols) ** 2)          # E[|y|²]
    c2 = np.mean(symbols ** 2)                   # E[y²]  (image component)

    if abs(c2) < 1e-10:
        return symbols   # no imbalance detected

    # Compensation matrix (2×2 equaliser)
    alpha = c2 / c1
    denom = 1.0 - abs(alpha) ** 2
    if denom < 1e-8:
        return symbols

    corrected = (symbols - alpha * np.conj(symbols)) / denom
    return corrected


def _doppler_correction(signal: np.ndarray,
                        sample_rate: float,
                        carrier_freq_hz: float,
                        coarse_freq_est_hz: float = None) -> np.ndarray:
    """
    Carrier-frequency-offset (CFO) correction.

    If coarse_freq_est_hz is not supplied, the offset is estimated from
    the 4th-power non-data-aided (NDA) estimator, which removes the QAM
    modulation for QPSK/QAM (raises to the 4th power to collapse the
    constellation, then looks for the spectral peak).
    """
    if coarse_freq_est_hz is None:
        # 4th-power CFO estimator
        s4 = signal ** 4
        spectrum = np.fft.fft(s4)
        freqs = np.fft.fftfreq(len(s4), d=1.0 / sample_rate)
        peak_idx = np.argmax(np.abs(spectrum))
        estimated_offset_hz = freqs[peak_idx] / 4.0  # divide back by 4
    else:
        estimated_offset_hz = coarse_freq_est_hz

    t = np.arange(len(signal)) / sample_rate
    corrected = signal * np.exp(-1j * 2 * np.pi * estimated_offset_hz * t)
    return corrected, estimated_offset_hz


# ---------------------------------------------------------------------------
# Main receiver function
# ---------------------------------------------------------------------------

@dataclass
class RxResult:
    """Container for receiver outputs."""
    symbols: np.ndarray           # demodulated symbols (post-matched-filter)
    rx_bits: np.ndarray           # hard-decision bits
    ber: float
    ser: float
    n_bit_errors: int
    estimated_doppler_hz: float = 0.0


def receive(rx_signal: np.ndarray,
            tx_bits: np.ndarray,
            cfg: Config,
            override_doppler_hz: float = None) -> RxResult:
    """
    Run the full receiver chain.

    Parameters
    ----------
    rx_signal          : complex oversampled signal from the channel
    tx_bits            : reference bit stream (for BER computation)
    cfg                : simulation Config
    override_doppler_hz: if given, use this value for correction instead of
                         the blind estimator (simulates perfect Doppler knowledge)

    Returns
    -------
    RxResult
    """
    sig = rx_signal.copy()

    # ------------------------------------------------------------------
    # 1. Doppler / CFO correction  (operate on oversampled signal)
    # ------------------------------------------------------------------
    estimated_doppler = 0.0
    if cfg.apply_doppler_correction:
        known_doppler = override_doppler_hz if override_doppler_hz is not None else None
        sig, estimated_doppler = _doppler_correction(
            sig,
            sample_rate=cfg.sample_rate_hz,
            carrier_freq_hz=cfg.carrier_freq_hz,
            coarse_freq_est_hz=known_doppler,
        )
        if cfg.verbose:
            print(f"  [Rx] CFO correction: estimated={estimated_doppler:.3f} Hz")

    # ------------------------------------------------------------------
    # 2. SRRC matched filter + downsample
    # ------------------------------------------------------------------
    delay = filter_delay(cfg.span, cfg.samples_per_symbol)
    symbols = rx_filter(sig, cfg.srrc_h if hasattr(cfg, 'srrc_h') else
                        _get_srrc_h(cfg),
                        cfg.samples_per_symbol, delay)

    # ------------------------------------------------------------------
    # 3. DC offset correction
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset and cfg.apply_dc_correction:
        symbols = _dc_correction(symbols)

    # ------------------------------------------------------------------
    # 4. AGC
    # ------------------------------------------------------------------
    if cfg.apply_agc:
        symbols = _agc(symbols, target_power=cfg.agc_target_power)

    # ------------------------------------------------------------------
    # 5. I/Q imbalance compensation
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance and cfg.apply_iq_correction:
        symbols = _iq_imbalance_correction(symbols)

    # ------------------------------------------------------------------
    # 6. Hard-decision 16-QAM demodulation → bits
    # ------------------------------------------------------------------
    rx_bits = symbols_to_bits(symbols, order=cfg.modulation_order)

    # ------------------------------------------------------------------
    # 7. BER / SER  (align lengths)
    # ------------------------------------------------------------------
    ber, ser, n_errors = symbol_error_rate(tx_bits, rx_bits, cfg.bits_per_symbol)

    if cfg.verbose:
        print(f"  [Rx] BER = {ber:.2e}  ({n_errors}/{min(len(tx_bits),len(rx_bits))} bit errors)")

    return RxResult(
        symbols=symbols,
        rx_bits=rx_bits,
        ber=ber,
        ser=ser,
        n_bit_errors=n_errors,
        estimated_doppler_hz=estimated_doppler,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_srrc_h(cfg: Config) -> np.ndarray:
    """Compute SRRC coefficients from config (avoid circular import)."""
    from filters import srrc_coeffs
    return srrc_coeffs(cfg.rolloff, cfg.span, cfg.samples_per_symbol)


def attach_srrc_h(cfg: Config) -> None:
    """Compute and cache SRRC coefficients on the config object."""
    cfg.srrc_h = _get_srrc_h(cfg)
