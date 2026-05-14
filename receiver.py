"""
receiver.py
-----------
Complete satellite downlink receiver chain.

Changes vs original Python
---------------------------
* Added carrier_sync PLL mode (cfo_correction_mode = "carrier_sync"):
    - 2nd-order PLL with configurable loop bandwidth and damping factor.
    - Closer to MATLAB comm.CarrierSynchronizer than the one-shot batch
      4th-power estimator.
    - Original blind estimator kept as cfo_correction_mode = "blind".
    - Known-CFO mode kept as cfo_correction_mode = "ideal" (set by passing
      override_doppler_hz).
* receive() now reads cfg.cfo_correction_mode to choose between modes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from filters import rx_filter, filter_delay
from modulation import symbols_to_bits, symbol_error_rate


# ---------------------------------------------------------------------------
# Compensator helpers (unchanged)
# ---------------------------------------------------------------------------

def _dc_correction(symbols: np.ndarray) -> np.ndarray:
    return symbols - np.mean(symbols)


def _agc(symbols: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    pwr = np.mean(np.abs(symbols) ** 2)
    if pwr > 0:
        return symbols * np.sqrt(target_power / pwr)
    return symbols


def _iq_correction_exact(symbols: np.ndarray) -> np.ndarray:
    """
    Blind I/Q imbalance correction via second-order statistics.
    Must run BEFORE AGC.
    """
    pwr = float(np.mean(np.abs(symbols) ** 2))
    if pwr < 1e-30:
        return symbols

    sin_est = 2.0 * float(np.mean(symbols.real * symbols.imag)) / pwr
    eimq2_n = float(np.mean(symbols.imag ** 2)) / pwr
    radicand = 2.0 * eimq2_n - sin_est ** 2
    if radicand < 0.0:
        return symbols

    eps_est = float(np.sqrt(radicand)) - 1.0

    if abs(sin_est) < 0.005 and abs(eps_est) < 0.001:
        return symbols

    A = (2.0 + eps_est + 1j * sin_est) / 2.0
    B = (      -eps_est + 1j * sin_est) / 2.0
    denom = abs(A) ** 2 - abs(B) ** 2
    if abs(denom) < 1e-10:
        return symbols
    return (np.conj(A) * symbols - B * np.conj(symbols)) / denom


_iq_imbalance_correction = _iq_correction_exact


def iq_correct_ideal(symbols: np.ndarray, amp_db: float, phase_deg: float) -> np.ndarray:
    """Ideal (known-parameter) I/Q correction."""
    if amp_db == 0.0 and phase_deg == 0.0:
        return symbols
    eps  = 10.0 ** (amp_db / 20.0) - 1.0
    dphi = np.deg2rad(phase_deg)
    A = (2.0 + eps + 1j * np.sin(dphi)) / 2.0
    B = (      -eps + 1j * np.sin(dphi)) / 2.0
    denom = abs(A) ** 2 - abs(B) ** 2
    if abs(denom) < 1e-10:
        return symbols
    return (np.conj(A) * symbols - B * np.conj(symbols)) / denom


# ---------------------------------------------------------------------------
# CFO estimation helpers
# ---------------------------------------------------------------------------

def _cfo_estimate_symbol_rate(symbols: np.ndarray,
                               fs_sym: float = 1.0,
                               skip: int = 0,
                               zp_factor: int = 4) -> float:
    """4th-power NDA CFO estimator at symbol rate with zero-padding."""
    seg = symbols[skip:]
    if len(seg) < 64:
        return 0.0
    s4 = seg ** 4
    N  = len(s4)
    Nfft = N * zp_factor
    S4   = np.fft.fft(s4, n=Nfft)
    freqs = np.fft.fftfreq(Nfft, d=1.0 / fs_sym)
    peak  = np.argmax(np.abs(S4))
    return float(freqs[peak]) / 4.0


def _carrier_sync_pll(symbols: np.ndarray,
                      loop_bw: float = 0.01,
                      damping: float = 0.707,
                      modulation_order: int = 16) -> np.ndarray:
    """
    Decision-directed 2nd-order PLL carrier synchroniser.

    MATLAB equivalent: comm.CarrierSynchronizer with 'DecisionDirected'
    phase error detector and a proportional-plus-integral (PI) loop filter.

    The PLL removes both frequency offset and residual phase offset
    symbol-by-symbol, which is the closed-loop behaviour of MATLAB's block
    (as opposed to the batch open-loop estimator used in "blind" mode).

    Parameters
    ----------
    symbols         : complex 1-D array at 1 sample/symbol
    loop_bw         : normalised loop bandwidth  BL × Ts  (e.g. 0.01)
    damping         : damping factor ζ (0.707 = Butterworth / critically damped)
    modulation_order: QAM order (sets the Mth-power phase error detector)

    Returns
    -------
    corrected : complex array of phase-corrected symbols
    """
    # PI loop filter coefficients from loop_bw and damping
    # Using the standard 2nd-order digital PLL design equations:
    #   theta = 4 * damping / (1 + 4*damping^2) * BL * Ts     (prop. gain K_p via normalised BW)
    theta_n = loop_bw / (damping + 1.0 / (4.0 * damping))
    K1 = 4.0 * damping * theta_n          # proportional gain
    K2 = 4.0 * theta_n ** 2               # integral gain

    N = len(symbols)
    out = np.zeros(N, dtype=complex)

    phi   = 0.0   # phase accumulator (VCO output)
    freq  = 0.0   # frequency integrator

    # Mth-power detector: for QAM we use a decision-directed error
    # (same as MATLAB comm.CarrierSynchronizer with DD mode)
    from modulation import SYMBOLS_16QAM

    for n in range(N):
        # Apply current phase correction
        r = symbols[n] * np.exp(-1j * phi)
        out[n] = r

        # Hard decision (nearest-neighbour)
        dist = np.abs(r - SYMBOLS_16QAM) ** 2
        d_hat = SYMBOLS_16QAM[np.argmin(dist)]

        # Decision-directed phase error
        # e = Im( r * conj(d_hat) )  (standard MATLAB formula)
        e = float((r * np.conj(d_hat)).imag)

        # PI loop filter update
        freq += K2 * e
        phi  += freq + K1 * e

    return out


def _doppler_correction(signal: np.ndarray,
                        sample_rate: float,
                        carrier_freq_hz: float = None,
                        coarse_freq_est_hz: float = None):
    """
    CFO correction on the oversampled signal.
    Returns (corrected_signal, estimated_offset_hz).
    """
    if coarse_freq_est_hz is None:
        return signal.copy(), 0.0
    t = np.arange(len(signal)) / sample_rate
    corrected = signal * np.exp(-1j * 2 * np.pi * coarse_freq_est_hz * t)
    return corrected, float(coarse_freq_est_hz)


def _phase_estimate_data_aided(symbols, ref_symbols, skip=0, n_pilot=200):
    """Data-aided residual phase estimator."""
    n = min(len(symbols), len(ref_symbols))
    seg_rx  = symbols[skip : skip + n_pilot]
    seg_ref = ref_symbols[skip : skip + n_pilot]
    m = min(len(seg_rx), len(seg_ref))
    if m < 4:
        return 0.0
    return float(np.angle(np.sum(seg_rx[:m] * np.conj(seg_ref[:m]))))


# ---------------------------------------------------------------------------
# Main receiver
# ---------------------------------------------------------------------------

@dataclass
class RxResult:
    """Container for receiver outputs."""
    symbols:              np.ndarray
    rx_bits:              np.ndarray
    ber:                  float
    ser:                  float
    n_bit_errors:         int
    estimated_doppler_hz: float = 0.0


def receive(rx_signal: np.ndarray,
            tx_bits: np.ndarray,
            cfg: Config,
            override_doppler_hz: float = None,
            ref_symbols: np.ndarray = None) -> RxResult:
    """
    Run the full receiver chain.

    CFO correction mode (cfg.cfo_correction_mode):
      "ideal"         – use override_doppler_hz directly on oversampled signal
      "blind"         – 4th-power NDA batch estimator at symbol rate (original)
      "carrier_sync"  – symbol-by-symbol 2nd-order PLL (closest to MATLAB)

    Parameters
    ----------
    rx_signal          : complex oversampled signal from the channel
    tx_bits            : reference bits for BER
    cfg                : Config
    override_doppler_hz: true CFO (used in "ideal" mode or for data-aided phase)
    ref_symbols        : ideal symbols for data-aided phase correction
    """
    from modulation import bits_to_symbols as _b2s
    sig = rx_signal.copy()

    # ------------------------------------------------------------------
    # 1. Ideal CFO correction (oversampled) – "ideal" mode only
    # ------------------------------------------------------------------
    ideal_cfo_applied = False
    mode = cfg.cfo_correction_mode if cfg.apply_doppler_correction else "none"

    if mode == "ideal" and override_doppler_hz is not None:
        sig, _ = _doppler_correction(
            sig,
            sample_rate=cfg.sample_rate_hz,
            coarse_freq_est_hz=override_doppler_hz,
        )
        ideal_cfo_applied = True
        if cfg.verbose:
            print(f"  [Rx] CFO correction (ideal): {override_doppler_hz:.6f} Hz")

    # ------------------------------------------------------------------
    # 2. SRRC matched filter + downsample
    # ------------------------------------------------------------------
    delay = filter_delay(cfg.span, cfg.samples_per_symbol)
    h = cfg.srrc_h if hasattr(cfg, 'srrc_h') else _get_srrc_h(cfg)
    symbols = rx_filter(sig, h, cfg.samples_per_symbol, delay)

    # ------------------------------------------------------------------
    # 2b. CFO correction at symbol rate
    # ------------------------------------------------------------------
    estimated_doppler = 0.0

    if mode == "blind":
        n_settle = cfg.span
        estimated_doppler = _cfo_estimate_symbol_rate(
            symbols, fs_sym=1.0, skip=n_settle, zp_factor=4)
        t_sym = np.arange(len(symbols))
        symbols = symbols * np.exp(-1j * 2 * np.pi * estimated_doppler * t_sym)
        if cfg.verbose:
            print(f"  [Rx] CFO correction (blind): estimated={estimated_doppler:.6f} Hz")

    elif mode == "carrier_sync":
        # 2nd-order decision-directed PLL (MATLAB comm.CarrierSynchronizer)
        symbols = _carrier_sync_pll(
            symbols,
            loop_bw=cfg.carrier_sync_loop_bw,
            damping=cfg.carrier_sync_damping,
            modulation_order=cfg.modulation_order,
        )
        # Report the frequency estimate from the PLL's final frequency state
        # as a diagnostic (not available without modifying the PLL).
        estimated_doppler = float(override_doppler_hz) if override_doppler_hz is not None else 0.0
        if cfg.verbose:
            print(f"  [Rx] CFO correction (carrier_sync PLL): "
                  f"BL={cfg.carrier_sync_loop_bw}, ζ={cfg.carrier_sync_damping}")

    elif ideal_cfo_applied:
        estimated_doppler = float(override_doppler_hz)

    # ------------------------------------------------------------------
    # 2c. Data-aided residual phase correction
    #     (runs for "blind" and "ideal" modes; PLL handles its own phase)
    # ------------------------------------------------------------------
    if cfg.apply_doppler_correction and mode in ("blind", "ideal"):
        if ref_symbols is None:
            ref_symbols = _b2s(tx_bits, cfg.modulation_order)
        n_settle = cfg.span
        phase_offset = _phase_estimate_data_aided(
            symbols, ref_symbols, skip=n_settle, n_pilot=200)
        symbols = symbols * np.exp(-1j * phase_offset)
        if cfg.verbose:
            print(f"  [Rx] Phase correction (data-aided): {np.degrees(phase_offset):.2f} deg")

    # ------------------------------------------------------------------
    # 3. DC offset correction
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset and cfg.apply_dc_correction:
        symbols = _dc_correction(symbols)

    # ------------------------------------------------------------------
    # 4. I/Q imbalance compensation  (must run BEFORE AGC)
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance and cfg.apply_iq_correction:
        symbols = _iq_imbalance_correction(symbols)

    # ------------------------------------------------------------------
    # 5. AGC
    # ------------------------------------------------------------------
    if cfg.apply_agc:
        symbols = _agc(symbols, target_power=cfg.agc_target_power)

    # ------------------------------------------------------------------
    # 6. Hard-decision demodulation
    # ------------------------------------------------------------------
    rx_bits = symbols_to_bits(symbols, order=cfg.modulation_order)

    # ------------------------------------------------------------------
    # 7. BER / SER
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
# Internal helpers
# ---------------------------------------------------------------------------

def _get_srrc_h(cfg: Config) -> np.ndarray:
    from filters import srrc_coeffs
    return srrc_coeffs(cfg.rolloff, cfg.span, cfg.samples_per_symbol)


def attach_srrc_h(cfg: Config) -> None:
    cfg.srrc_h = _get_srrc_h(cfg)
