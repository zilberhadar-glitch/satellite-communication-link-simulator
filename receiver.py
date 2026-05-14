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


def _iq_correction_exact(symbols: np.ndarray) -> np.ndarray:
    """
    Model-matched I/Q imbalance correction.

    The impairment model in impairments.apply_iq_imbalance() is:
        I_out = I_in
        Q_out = (1 + eps) * Q_in  +  I_in * sin(dphi)

    This is algebraically equivalent to:
        y = A * x  +  B * conj(x)
    with
        A = (2 + eps + j*sin(dphi)) / 2
        B = (  -eps + j*sin(dphi)) / 2

    Parameter estimation via cross-moments
    ----------------------------------------
    Because I_out = I_in, the I rail is untouched. This gives:

        E[Re(y) * Im(y)] = E[I_in * Q_out]
                         = E[I * ((1+eps)*Q + I*sin)]
                         = (1+eps)*E[I*Q]  +  sin*E[I^2]
                         = 0  +  sin * 0.5        (balanced 16-QAM: E[I*Q]=0)
        => sin(dphi) = 2 * E[Re(y) * Im(y)]

        E[Im(y)^2] = E[((1+eps)*Q + I*sin)^2]
                   = (1+eps)^2 * 0.5  +  sin^2 * 0.5
        => eps = sqrt(2*E[Im(y)^2] - sin(dphi)^2) - 1

    Correction (exact inverse)
    --------------------------
        x = (conj(A) * y  -  B * conj(y)) / (|A|^2 - |B|^2)

    Important: this function must be called BEFORE AGC.
    AGC rescales amplitude uniformly, which alters E[Im^2] and would
    break the eps estimate. The sin estimate is AGC-invariant in theory
    but the eps estimate is not, so always correct before AGC.

    Detection guard
    ---------------
    If both |sin_est| < 0.005 and |eps_est| < 0.001 the signal
    appears balanced; skip correction to avoid injecting finite-sample
    noise into a clean signal.
    """
    # Normalise by average symbol power so estimates are scale-invariant.
    # After the matched filter, symbols are at physical signal amplitude
    # (~10^-7 after path loss), not normalised to 1. Without this division
    # the cross-moments are on the order of 10^-15 and eps_est collapses to -1.
    pwr = float(np.mean(np.abs(symbols) ** 2))
    if pwr < 1e-30:
        return symbols

    # Estimate sin(dphi) from the normalised I-Q cross-moment
    sin_est = 2.0 * float(np.mean(symbols.real * symbols.imag)) / pwr

    # Estimate eps from the normalised Q-rail power
    eimq2_n = float(np.mean(symbols.imag ** 2)) / pwr
    radicand = 2.0 * eimq2_n - sin_est ** 2
    if radicand < 0.0:
        return symbols      # numerical issue; leave symbols unchanged

    eps_est = float(np.sqrt(radicand)) - 1.0

    # Detection guard — skip when both parameters are below noise floor
    if abs(sin_est) < 0.005 and abs(eps_est) < 0.001:
        return symbols

    # Reconstruct A and B from the estimated parameters
    A = (2.0 + eps_est + 1j * sin_est) / 2.0
    B = (      -eps_est + 1j * sin_est) / 2.0

    denom = abs(A) ** 2 - abs(B) ** 2
    if abs(denom) < 1e-10:
        return symbols

    return (np.conj(A) * symbols - B * np.conj(symbols)) / denom


# Public alias — nothing outside this file needs to change.
_iq_imbalance_correction = _iq_correction_exact


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
    # 4. I/Q imbalance compensation  (must run BEFORE AGC)
    #
    #    The moment-matched estimator uses the ratio of I and Q power,
    #    which is preserved by the matched filter but would be altered
    #    by AGC's uniform amplitude scaling.  Correcting here, before
    #    AGC normalises the constellation, gives the estimator access to
    #    the unscaled second-order statistics it needs.
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance and cfg.apply_iq_correction:
        symbols = _iq_imbalance_correction(symbols)

    # ------------------------------------------------------------------
    # 5. AGC
    # ------------------------------------------------------------------
    if cfg.apply_agc:
        symbols = _agc(symbols, target_power=cfg.agc_target_power)

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
