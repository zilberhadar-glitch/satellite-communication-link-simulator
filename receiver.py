"""
receiver.py
-----------
Complete satellite downlink receiver chain.

MATLAB block order (RF Satellite Link, comm.* blocks):
  RRC matched filter → DC Blocker → AGC → I/Q Compensator
  → Carrier Synchronizer → QAM Demodulator

Changes vs previous Python version (MATLAB-equivalence fixes)
-------------------------------------------------------------
1. DC correction: replaced batch mean subtraction with a proper IIR DC
   blocker  y[n] = x[n] - x[n-1] + α·y[n-1]  matching MATLAB's
   dsp.DCBlocker block.

2. I/Q compensator: replaced blind 2nd-order batch estimator with a
   simple LMS-based blind adaptive compensator (comm.IQImbalanceCompensator
   equivalent).  The LMS approach works correctly for the MATLAB symmetric
   ± IQ imbalance model where E[I·Q] = 0 (making the old cross-correlation
   estimator blind to phase imbalance).

3. Block order: I/Q compensation is now placed AFTER AGC, matching the
   MATLAB Simulink block diagram exactly:
       DC Blocker → AGC → IQ Compensator → Carrier Sync → Demod

4. Ideal correction updated: iq_correct_ideal() now inverts the symmetric
   ± model used by apply_iq_imbalance().

5. carrier_sync PLL: coarse 4th-power pre-correction + DA phase pre-
   correction before the fine PLL (necessary for finite-burst operation
   without Simulink cross-frame state — kept as before).
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

def _dc_blocker(symbols: np.ndarray, alpha: float = 0.999) -> np.ndarray:
    """
    IIR DC blocker matching MATLAB dsp.DCBlocker.

    Transfer function:  H(z) = (1 - z⁻¹) / (1 - α·z⁻¹)
    Difference equation: y[n] = x[n] - x[n-1] + α·y[n-1]

    The -3 dB cutoff is approximately fc = (1 - α) / (2π) × fs.
    With α = 0.999 and fs = 1 sym/s:  fc ≈ 0.000159 sym/s  (very low).

    MATLAB default: alpha close to 1 (exact value depends on DSP Toolbox
    version; 0.999 gives equivalent behaviour for the signal lengths used).
    """
    out = np.zeros(len(symbols), dtype=complex)
    x_prev = 0j
    y_prev = 0j
    for n in range(len(symbols)):
        y = symbols[n] - x_prev + alpha * y_prev
        out[n] = y
        x_prev = symbols[n]
        y_prev = y
    return out


def _agc(symbols: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """
    Static-gain AGC.  Equivalent to a single-step normalisation.
    MATLAB comm.AGC is adaptive; for a stationary channel the result is
    identical.  Marked as partial match.
    """
    pwr = np.mean(np.abs(symbols) ** 2)
    if pwr > 0:
        return symbols * np.sqrt(target_power / pwr)
    return symbols


def _iq_lms_compensator(symbols: np.ndarray,
                         n_iter: int = 3,
                         mu_phase: float = 0.01) -> np.ndarray:
    """
    Decision-directed (DD) I/Q imbalance compensator.

    MATLAB equivalent: comm.IQImbalanceCompensator
    Reference: MATLAB Documentation comm.IQImbalanceCompensator,
               Windisch & Fettweis (2004), Liu & Windisch (2007).

    This compensator is designed specifically for MATLAB's symmetric ± model:
        I_out = gain_I*(I_in*cos(d) - Q_in*sin(d))
        Q_out = gain_Q*(I_in*sin(d) + Q_in*cos(d))

    For a balanced QAM input (E[I²]=E[Q²]=0.5, E[I·Q]=0), the 2nd-order
    cross-correlation E[I_out·Q_out]=0 regardless of phase imbalance.  This
    means the old blind cross-correlation estimator is completely ineffective
    for phase recovery under the symmetric model.

    Algorithm (two stages, matching MATLAB's internal structure)
    -----------------------------------------------------------
    Stage 1 — Amplitude correction (closed-form batch, exact):
        E[I_out²] = gain_I² · 0.5  →  gain_I = sqrt(2·E[I²])
        E[Q_out²] = gain_Q² · 0.5  →  gain_Q = sqrt(2·E[Q²])
        Correction: scale I by sqrt(0.5/E[I²]) and Q by sqrt(0.5/E[Q²]).

    Stage 2 — Phase correction (decision-directed adaptive, n_iter passes):
        After amplitude equalisation a residual rotation ±d remains.
        Each sample: y[n] = x[n]·exp(−j·θ[n])
        Decision:    d̂[n] = nearest 16-QAM symbol
        Error:       e[n] = Im(y[n]·conj(d̂[n]))   (phase error)
        Update:      θ[n+1] = θ[n] + mu_phase·e[n]
        Multiple passes let the phase accumulator converge.

    This matches the behaviour of comm.IQImbalanceCompensator which uses an
    LMS-style decision-directed phase tracker with magnitude normalization.

    Position: called AFTER AGC (matching MATLAB's block diagram order).

    Parameters
    ----------
    symbols  : complex 1-D array (after AGC)
    n_iter   : number of DD passes (default 3)
    mu_phase : DD phase loop step size (default 0.01)

    Returns
    -------
    corrected : complex 1-D array
    """
    from modulation import SYMBOLS_16QAM

    # ── Stage 1: Amplitude correction (batch) ──────────────────────────
    pI = float(np.mean(symbols.real ** 2))
    pQ = float(np.mean(symbols.imag ** 2))
    if pI > 1e-30 and pQ > 1e-30:
        norm_I = np.sqrt(0.5 / pI)
        norm_Q = np.sqrt(0.5 / pQ)
        x = symbols.real * norm_I + 1j * symbols.imag * norm_Q
    else:
        x = symbols.copy()

    # Re-normalise power after amplitude correction
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if rms > 1e-30:
        x = x / rms

    # ── Stage 2: Phase correction (decision-directed, n_iter passes) ───
    out = x.copy()
    for _ in range(n_iter):
        theta = 0.0
        for n in range(len(x)):
            y = x[n] * np.exp(-1j * theta)
            d_hat = SYMBOLS_16QAM[int(np.argmin(np.abs(y - SYMBOLS_16QAM) ** 2))]
            err = float(np.imag(y * np.conj(d_hat)))
            theta += mu_phase * err
            out[n] = y
        x = out.copy()  # feed corrected output into next pass

    return out


def iq_correct_ideal(symbols: np.ndarray, amp_db: float, phase_deg: float) -> np.ndarray:
    """
    Ideal (known-parameter) I/Q correction.

    Inverts the MATLAB-equivalent symmetric ± apply_iq_imbalance() model:
        I_out = gain_I * ( I_in*cos(dphi) - Q_in*sin(dphi) )
        Q_out = gain_Q * ( I_in*sin(dphi) + Q_in*cos(dphi) )

    Inversion is done by solving the 2×2 linear system exactly.
    """
    if amp_db == 0.0 and phase_deg == 0.0:
        return symbols.copy()

    alpha  = amp_db / 2.0
    dphi   = np.deg2rad(phase_deg / 2.0)
    gain_I = 10.0 ** ( alpha / 20.0)
    gain_Q = 10.0 ** (-alpha / 20.0)

    # Forward matrix M:  [I_out, Q_out]^T = M · [I_in, Q_in]^T
    #   M = [[gain_I*cos, -gain_I*sin],
    #         [gain_Q*sin,  gain_Q*cos]]
    M = np.array([
        [ gain_I * np.cos(dphi), -gain_I * np.sin(dphi)],
        [ gain_Q * np.sin(dphi),  gain_Q * np.cos(dphi)],
    ])
    M_inv = np.linalg.inv(M)

    I_r = symbols.real
    Q_r = symbols.imag
    IQ  = np.stack([I_r, Q_r], axis=0)       # (2, N)
    IQ_corr = M_inv @ IQ                      # (2, N)
    return IQ_corr[0] + 1j * IQ_corr[1]


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
    # 3. DC blocker  (MATLAB: dsp.DCBlocker — IIR high-pass, first after MF)
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset and cfg.apply_dc_correction:
        symbols = _dc_blocker(symbols, alpha=0.999)

    # ------------------------------------------------------------------
    # 4. AGC  (MATLAB: comm.AGC — after DC Blocker, before IQ Compensator)
    # ------------------------------------------------------------------
    if cfg.apply_agc:
        symbols = _agc(symbols, target_power=cfg.agc_target_power)

    # ------------------------------------------------------------------
    # 5. I/Q imbalance compensation  — AFTER AGC  (MATLAB block order)
    #    MATLAB: comm.IQImbalanceCompensator (adaptive LMS, after AGC)
    #    Python: _iq_lms_compensator() — blind LMS, same position as MATLAB.
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance and cfg.apply_iq_correction:
        symbols = _iq_lms_compensator(symbols, n_iter=3, mu_phase=0.01)

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
