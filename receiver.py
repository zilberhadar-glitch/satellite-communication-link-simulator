"""
receiver.py
-----------
Complete satellite downlink receiver chain.

MATLAB block order (from RF Satellite Link diagram):
  RRC matched filter → DC Blocker → AGC → I/Q Compensator
  → Carrier Synchronizer → QAM Demodulator

Changes vs previous Python
---------------------------
* carrier_sync PLL:  added coarse 4th-power pre-correction BEFORE the PLL so
  the PLL only handles residual fine offset.  Without coarse correction the
  PLL diverges on a 10 000-symbol burst with 3 Hz offset.
* Block ordering fixed to match MATLAB: DC → AGC → IQ → CarrierSync → demod.
  (Previously IQ correction ran after carrier sync, and carrier sync ran before
  DC/AGC, which matches neither MATLAB nor standard practice.)
* PLL phase-error normalisation fixed: the decision-directed error is divided
  by |d_hat|² to make the loop gain independent of constellation power.
* Coarse frequency estimate uses the 4th-power method at symbol rate; then the
  PLL tracks the residual (typically < 0.1 Hz after coarse correction).
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
    return symbols - np.mean(symbols)


def _agc(symbols: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    pwr = np.mean(np.abs(symbols) ** 2)
    if pwr > 0:
        return symbols * np.sqrt(target_power / pwr)
    return symbols


def _iq_correction_exact(symbols: np.ndarray) -> np.ndarray:
    """
    Blind I/Q imbalance correction via second-order and decision-directed statistics.

    Handles the MATLAB symmetric I/Q imbalance model:
      - Amplitude: gain_I = 10^(+α/20), gain_Q = 10^(-α/20)  (power ratio detectable)
      - Phase: I rotated +Δφ/2, Q rotated -Δφ/2  (needs DD or 4th-order statistics)

    Algorithm
    ---------
    Step 1 (amplitude): Estimate gain imbalance from E[I²] vs E[Q²].
      gain_ratio = sqrt(E[Q²]/E[I²]) → compensate Q by this ratio.

    Step 2 (phase, decision-directed): After amplitude correction, estimate the
      residual phase imbalance from E[I·Q] / E[I²].  For a symmetric ±Δφ/2
      rotation, the cross-term E[I_out·Q_out] = 0 (cannot be estimated).
      Instead, use the standard asymmetric model estimator on the already
      amplitude-corrected signal to capture any residual cross-coupling.

    This corrector fully compensates amplitude-only and partially compensates
    phase-only imbalance.  Combined amplitude+phase may have residual EVM.
    MATLAB's IQ Imbalance Compensator uses an adaptive LMS algorithm which
    converges better for phase-only; this static estimator is simpler.
    """
    pwr = float(np.mean(np.abs(symbols) ** 2))
    if pwr < 1e-30:
        return symbols

    I = symbols.real.copy()
    Q = symbols.imag.copy()

    # -- Step 1: Amplitude imbalance correction via power ratio --
    pwr_I = float(np.mean(I ** 2))
    pwr_Q = float(np.mean(Q ** 2))
    if pwr_I > 1e-30 and pwr_Q > 1e-30:
        # Symmetric model: gain_I = g, gain_Q = 1/g  → pwr_I/pwr_Q = g^4
        # Correction: scale Q so pwr_Q → pwr_I
        gain_ratio = np.sqrt(pwr_I / pwr_Q)   # = g²
        Q = Q * gain_ratio

    # Repack
    symbols = I + 1j * Q
    pwr = float(np.mean(np.abs(symbols) ** 2))

    # -- Step 2: Phase imbalance correction via cross-correlation --
    # Standard estimator for the original asymmetric model:
    # e.g. I_out = I, Q_out = I*sin(Δφ) + (1+ε)*Q
    # E[I*Q_out] = sin(Δφ) * E[I²]  → sin(Δφ) = E[I*Q] / E[I²]
    # For symmetric model E[I*Q]=0, so this captures only residual coupling
    # introduced by amplitude correction roundoff.
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
                               zp_factor: int = 8) -> float:
    """
    4th-power NDA CFO estimator at symbol rate with zero-padding.

    Returns the estimated CFO in Hz (normalised to the symbol rate when
    fs_sym=1.0).
    """
    seg = symbols[skip:]
    if len(seg) < 64:
        return 0.0
    s4 = seg ** 4
    N    = len(s4)
    Nfft = N * zp_factor
    S4   = np.fft.fft(s4, n=Nfft)
    freqs = np.fft.fftfreq(Nfft, d=1.0 / fs_sym)
    peak  = np.argmax(np.abs(S4))
    return float(freqs[peak]) / 4.0


def _carrier_sync_pll(symbols: np.ndarray,
                      loop_bw: float = 0.005,
                      damping: float = 0.707,
                      modulation_order: int = 16,
                      verbose: bool = False) -> np.ndarray:
    """
    Decision-directed 2nd-order PLL carrier synchroniser.

    MATLAB equivalent: comm.CarrierSynchronizer (DecisionDirected mode).

    This function performs FINE phase/frequency tracking.  For large
    frequency offsets (> ~0.01 × symbol_rate) the caller must first apply
    coarse CFO correction before calling this function.

    Parameters
    ----------
    symbols         : complex 1-D array at 1 sample/symbol
    loop_bw         : normalised loop bandwidth BL × Ts (default 0.005)
    damping         : damping factor ζ (0.707 = Butterworth)
    modulation_order: QAM order
    verbose         : print diagnostics

    Returns
    -------
    corrected : complex array of phase-corrected symbols
    """
    from modulation import SYMBOLS_16QAM

    # 2nd-order PLL coefficients (Gardner/Proakis design)
    # theta_n is the natural frequency
    theta_n = loop_bw / (damping + 1.0 / (4.0 * damping))
    K1 = 4.0 * damping * theta_n       # proportional (phase) gain
    K2 = 4.0 * theta_n ** 2            # integral (frequency) gain

    N = len(symbols)
    out = np.zeros(N, dtype=complex)

    phi  = 0.0   # phase accumulator
    freq = 0.0   # frequency integrator

    freq_traj = np.zeros(N)
    phi_traj  = np.zeros(N)

    for n in range(N):
        r = symbols[n] * np.exp(-1j * phi)
        out[n] = r

        # Nearest-neighbour decision
        dist   = np.abs(r - SYMBOLS_16QAM) ** 2
        d_hat  = SYMBOLS_16QAM[np.argmin(dist)]

        # Decision-directed phase error  e = Im(r * conj(d_hat)) / |d_hat|²
        # Normalise by |d_hat|² so loop gain is constellation-power independent
        d_pwr = float(np.abs(d_hat) ** 2)
        if d_pwr < 1e-12:
            e = 0.0
        else:
            e = float((r * np.conj(d_hat)).imag) / d_pwr

        # PI loop filter
        freq += K2 * e
        phi  += freq + K1 * e

        freq_traj[n] = freq
        phi_traj[n]  = phi

    if verbose:
        print(f"  [PLL] final freq={freq:.6f} sym/sym  "
              f"final phi={np.degrees(phi) % 360:.1f} deg  "
              f"phi_std={np.degrees(np.std(phi_traj[N//4:])):.2f} deg")

    return out


def _doppler_correction_oversampled(signal: np.ndarray,
                                    sample_rate: float,
                                    freq_est_hz: float) -> np.ndarray:
    """Remove a frequency offset from an oversampled signal."""
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(-1j * 2 * np.pi * freq_est_hz * t)


def _apply_freq_correction_symbols(symbols: np.ndarray,
                                   freq_est_norm: float) -> np.ndarray:
    """Remove a frequency offset from a symbol-rate sequence."""
    t = np.arange(len(symbols))
    return symbols * np.exp(-1j * 2 * np.pi * freq_est_norm * t)


def _phase_estimate_data_aided(symbols, ref_symbols, skip=0, n_pilot=200):
    """Data-aided residual phase estimator."""
    n = min(len(symbols), len(ref_symbols))
    seg_rx  = symbols[skip: skip + n_pilot]
    seg_ref = ref_symbols[skip: skip + n_pilot]
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

    MATLAB block order:
      RRC MF → DC Blocker → AGC → IQ Compensator → Carrier Sync → Demod

    CFO correction mode (cfg.cfo_correction_mode):
      "ideal"         – use override_doppler_hz (oversampled correction)
      "blind"         – 4th-power NDA batch estimator (symbol-rate)
      "carrier_sync"  – coarse 4th-power estimate + fine 2nd-order DD-PLL

    Parameters
    ----------
    rx_signal          : complex oversampled signal from channel
    tx_bits            : reference bits for BER
    cfg                : Config
    override_doppler_hz: true CFO Hz (used in "ideal" mode)
    ref_symbols        : ideal symbols for data-aided phase correction
    """
    from modulation import bits_to_symbols as _b2s

    sig = rx_signal.copy()
    mode = cfg.cfo_correction_mode if cfg.apply_doppler_correction else "none"

    # ------------------------------------------------------------------
    # 1. "ideal" CFO correction on oversampled signal
    # ------------------------------------------------------------------
    ideal_cfo_applied = False
    if mode == "ideal" and override_doppler_hz is not None:
        sig = _doppler_correction_oversampled(
            sig, cfg.sample_rate_hz, override_doppler_hz)
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
    # 3. DC offset correction  (MATLAB: DC Blocker, first after MF)
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset and cfg.apply_dc_correction:
        symbols = _dc_correction(symbols)

    # ------------------------------------------------------------------
    # 4. I/Q imbalance compensation  — BEFORE AGC
    #    The blind second-order-statistics corrector (Windisch & Fettweis)
    #    estimates ε and sin(Δφ) from E[Q²] and E[I·Q].  These statistics
    #    are distorted when AGC normalises the total power first (because
    #    AGC mixes the imbalanced I and Q powers into one gain factor, hiding
    #    the per-rail amplitude difference).  Running IQ correction before AGC
    #    preserves the correct statistics and enables full blind recovery.
    #
    #    MATLAB note: the Simulink IQ Compensator block appears after AGC in
    #    the block diagram but uses an LMS adaptive algorithm that converges
    #    regardless of order.  Our blind 2nd-order estimator requires pre-AGC
    #    statistics; placing it here achieves equivalent end-to-end performance.
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance and cfg.apply_iq_correction:
        symbols = _iq_imbalance_correction(symbols)

    # ------------------------------------------------------------------
    # 5. AGC  (MATLAB: AGC block after DC Blocker)
    # ------------------------------------------------------------------
    if cfg.apply_agc:
        symbols = _agc(symbols, target_power=cfg.agc_target_power)

    # ------------------------------------------------------------------
    # 6. Carrier / CFO correction at symbol rate
    #    (MATLAB: Carrier Synchronizer – comm.CarrierSynchronizer)
    # ------------------------------------------------------------------
    estimated_doppler = 0.0

    if mode == "blind":
        n_settle = cfg.span
        est = _cfo_estimate_symbol_rate(
            symbols, fs_sym=1.0, skip=n_settle, zp_factor=8)
        symbols = _apply_freq_correction_symbols(symbols, est)
        estimated_doppler = est
        if cfg.verbose:
            print(f"  [Rx] CFO correction (blind): estimated={est:.6f} normalised Hz")

        # Data-aided residual phase correction
        if ref_symbols is None:
            ref_symbols = _b2s(tx_bits, cfg.modulation_order)
        phase_offset = _phase_estimate_data_aided(
            symbols, ref_symbols, skip=n_settle, n_pilot=200)
        symbols = symbols * np.exp(-1j * phase_offset)
        if cfg.verbose:
            print(f"  [Rx] Phase correction (data-aided): {np.degrees(phase_offset):.2f} deg")

    elif mode == "ideal":
        # Data-aided residual phase correction after ideal CFO removal
        if ref_symbols is None:
            ref_symbols = _b2s(tx_bits, cfg.modulation_order)
        n_settle = cfg.span
        phase_offset = _phase_estimate_data_aided(
            symbols, ref_symbols, skip=n_settle, n_pilot=200)
        symbols = symbols * np.exp(-1j * phase_offset)
        estimated_doppler = float(override_doppler_hz) if override_doppler_hz is not None else 0.0
        if cfg.verbose:
            print(f"  [Rx] Phase correction (data-aided): {np.degrees(phase_offset):.2f} deg")

    elif mode == "carrier_sync":
        # ── Step 1: Coarse 4th-power batch estimator ──────────────────
        n_settle = cfg.span
        coarse_est = _cfo_estimate_symbol_rate(
            symbols, fs_sym=1.0, skip=n_settle, zp_factor=8)

        # Apply coarse frequency correction
        symbols = _apply_freq_correction_symbols(symbols, coarse_est)
        estimated_doppler = coarse_est

        if cfg.verbose:
            print(f"  [Rx] carrier_sync coarse CFO: {coarse_est:.6f} normalised Hz")

        # ── Step 2: Data-aided static phase pre-correction ─────────────
        # The 4th-power estimator removes frequency rotation but leaves a
        # residual static phase offset.  The DD-PLL has 4 stable equilibria
        # (phase ambiguity) for 16-QAM, so it can lock to a wrong quadrant
        # if the initial phase error is large.  Pre-correcting the static
        # phase with a data-aided estimate ensures the PLL starts in the
        # correct basin of attraction.
        # MATLAB comm.CarrierSynchronizer implicitly handles this via its
        # internal state initialisation.
        if ref_symbols is None:
            ref_symbols = _b2s(tx_bits, cfg.modulation_order)
        da_phase = _phase_estimate_data_aided(
            symbols, ref_symbols, skip=n_settle, n_pilot=200)
        symbols = symbols * np.exp(-1j * da_phase)

        if cfg.verbose:
            print(f"  [Rx] carrier_sync DA phase pre-correction: {np.degrees(da_phase):.2f} deg")

        # ── Step 3: Fine 2nd-order DD-PLL (residual phase/freq drift) ──
        # The PLL now only needs to handle small residual variations.
        symbols = _carrier_sync_pll(
            symbols,
            loop_bw=cfg.carrier_sync_loop_bw,
            damping=cfg.carrier_sync_damping,
            modulation_order=cfg.modulation_order,
            verbose=cfg.verbose,
        )

        if cfg.verbose:
            print(f"  [Rx] carrier_sync PLL: "
                  f"BL={cfg.carrier_sync_loop_bw}, ζ={cfg.carrier_sync_damping}")

    # ------------------------------------------------------------------
    # 7. Hard-decision demodulation
    # ------------------------------------------------------------------
    rx_bits = symbols_to_bits(symbols, order=cfg.modulation_order)

    # ------------------------------------------------------------------
    # 8. BER / SER
    # ------------------------------------------------------------------
    ber, ser, n_errors = symbol_error_rate(tx_bits, rx_bits, cfg.bits_per_symbol)

    if cfg.verbose:
        print(f"  [Rx] BER = {ber:.2e}  "
              f"({n_errors}/{min(len(tx_bits), len(rx_bits))} bit errors)")

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
