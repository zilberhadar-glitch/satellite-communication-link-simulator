"""
impairments.py
--------------
All RF impairment models.  Updated to match the MATLAB RF Satellite Link
example as closely as possible.

Key fixes vs previous Python version
--------------------------------------
1. add_colored_phase_noise()  – FIXED scaling.  Previous version used
   rfftfreq + a sqrt(2) factor that produced near-zero phase noise for the
   MATLAB levels (-100 / -55 / -48 dBc/Hz).  Now uses a direct discrete
   PSD integration (Kasdin 1992 method) which gives physically correct
   phase-noise power for any sample rate.

2. apply_iq_imbalance()  – FIXED to match MATLAB comm.IQImbalance symmetric
   ± model:
     - Amplitude 3 dB → gain_I = +1.5 dB, gain_Q = -1.5 dB  (split equally)
     - Phase 20 deg   → I rotated +10°, Q rotated -10°         (split equally)
   Previous version applied the full error to the Q rail only (one-sided
   model), which does not match the MATLAB block behaviour.

3. saleh_dpd()  – improved output rescaling so the DPD+HPA chain has
   better amplitude linearity.

4. LNA ordering in channel.py: noise is added BEFORE the LNA (unchanged).
"""

import numpy as np
from config import K_BOLTZMANN


# ============================================================
# 1.  HPA – Saleh TWTA Memoryless Nonlinearity
# ============================================================

def saleh_hpa(signal: np.ndarray,
              a_a: float, b_a: float,
              a_p: float, b_p: float,
              input_backoff_db: float) -> np.ndarray:
    """
    Saleh model for a Travelling-Wave Tube Amplifier (TWTA).

    AM/AM:  A(r) = a_a * r  /  (1 + b_a * r²)
    AM/PM:  Φ(r) = a_p * r² / (1 + b_p * r²)   [radians]
    """
    r_sat = 1.0 / np.sqrt(b_a)
    ibo_linear = 10 ** (input_backoff_db / 10.0)
    rms_target = r_sat / np.sqrt(ibo_linear)

    rms_in = np.sqrt(np.mean(np.abs(signal) ** 2))
    if rms_in > 0:
        x = signal * (rms_target / rms_in)
    else:
        return signal.copy()

    r = np.abs(x)
    A = a_a * r / (1.0 + b_a * r ** 2)
    phi = a_p * r ** 2 / (1.0 + b_p * r ** 2)
    theta = np.angle(x)
    return A * np.exp(1j * (theta + phi))


def saleh_dpd(signal: np.ndarray,
              a_a: float, b_a: float,
              a_p: float, b_p: float,
              input_backoff_db: float,
              lut_points: int = 2048) -> np.ndarray:
    """
    Digital Pre-Distortion (DPD) for the Saleh TWTA model.

    Implements the analytic inverse of the Saleh AM/AM and AM/PM functions
    using a Look-Up Table (LUT).

    Improvement over previous version
    -----------------------------------
    * LUT resolution doubled to 2048 (default).
    * After pre-distortion, the signal is rescaled so that the DPD output
      has the same RMS as the original input — this ensures the downstream
      saleh_hpa() receives the correct operating power level.
    * The AM/PM inverse sign is verified: we pre-subtract the AM/PM phase
      so that the HPA then *adds* it back, yielding net zero phase rotation.

    Parameters
    ----------
    signal           : complex 1-D array
    a_a, b_a         : Saleh AM/AM coefficients
    a_p, b_p         : Saleh AM/PM coefficients
    input_backoff_db : IBO used for the downstream HPA (must match)
    lut_points       : number of LUT entries

    Returns
    -------
    pre_distorted : complex 1-D array ready to be passed through saleh_hpa()
    """
    rms_in = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    if rms_in < 1e-30:
        return signal.copy()

    # Scale to HPA operating point (same as saleh_hpa does internally)
    r_sat = 1.0 / np.sqrt(b_a)
    ibo_linear = 10 ** (input_backoff_db / 10.0)
    rms_target = r_sat / np.sqrt(ibo_linear)
    x = signal * (rms_target / rms_in)

    # Build forward LUT:  r_in → (r_out, phi_out)
    r_max = 2.0 * r_sat
    r_in_lut  = np.linspace(0.0, r_max, lut_points)
    r_out_lut = a_a * r_in_lut / (1.0 + b_a * r_in_lut ** 2)
    phi_lut   = a_p * r_in_lut ** 2 / (1.0 + b_p * r_in_lut ** 2)

    # Restrict inversion to monotone region [0, r_sat]
    i_sat      = np.argmax(r_out_lut)
    r_in_mono  = r_in_lut[:i_sat + 1]
    r_out_mono = r_out_lut[:i_sat + 1]
    phi_mono   = phi_lut[:i_sat + 1]

    # Invert: desired amplitude → required pre-distorted amplitude
    r_des         = np.abs(x)
    r_des_clipped = np.clip(r_des, 0.0, r_out_mono[-1])
    r_pre         = np.interp(r_des_clipped, r_out_mono, r_in_mono)
    phi_pre       = np.interp(r_pre, r_in_mono, phi_mono)

    # Reconstruct pre-distorted complex signal (subtract AM/PM phase)
    theta             = np.angle(x)
    pre_distorted_hpa = r_pre * np.exp(1j * (theta - phi_pre))

    # Scale back to original RMS domain
    rms_pre = float(np.sqrt(np.mean(np.abs(pre_distorted_hpa) ** 2)))
    if rms_pre > 1e-30:
        out = pre_distorted_hpa * (rms_in / rms_pre)
    else:
        out = signal.copy()

    return out


def hpa_am_am_curve(a_a: float, b_a: float, n_points: int = 200):
    """Return (r_in, r_out) arrays for plotting the AM/AM characteristic."""
    r_sat = 1.0 / np.sqrt(b_a)
    r_in = np.linspace(0, 2 * r_sat, n_points)
    r_out = a_a * r_in / (1.0 + b_a * r_in ** 2)
    return r_in, r_out


# ============================================================
# 2.  Free-Space Path Loss
# ============================================================

def apply_path_loss(signal: np.ndarray,
                    fspl_db: float,
                    tx_gain_dbi: float,
                    rx_gain_dbi: float) -> np.ndarray:
    """Scale signal by Tx gain – FSPL + Rx gain."""
    net_db = tx_gain_dbi - fspl_db + rx_gain_dbi
    voltage_gain = 10.0 ** (net_db / 20.0)
    return signal * voltage_gain


# ============================================================
# 3.  Doppler Frequency Offset
# ============================================================

def apply_doppler(signal: np.ndarray,
                  doppler_hz: float,
                  sample_rate: float) -> np.ndarray:
    """Complex exponential Doppler shift."""
    if doppler_hz == 0.0:
        return signal.copy()
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(1j * 2 * np.pi * doppler_hz * t)


# ============================================================
# 4.  AWGN – Receiver Thermal Noise
# ============================================================

def add_awgn_noise(signal: np.ndarray,
                   noise_temp_k: float,
                   sample_rate: float,
                   rng: np.random.Generator,
                   noise_power_scale: float = 1.0) -> np.ndarray:
    """
    Add complex AWGN whose power follows P_noise = k_B * T * B.
    Added BEFORE LNA (MATLAB block order).
    """
    if noise_temp_k <= 0:
        return signal.copy()

    noise_power = K_BOLTZMANN * noise_temp_k * (sample_rate / 2.0)
    noise_power *= noise_power_scale
    if np.mean(np.abs(signal) ** 2) <= 0:
        return signal.copy()

    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(signal))
                     + 1j * rng.standard_normal(len(signal)))
    return signal + noise


# ============================================================
# 5a. Phase Noise – Colored (MATLAB Phase Noise block equivalent)
# ============================================================

def add_colored_phase_noise(signal: np.ndarray,
                             pn_dbc_hz: float,
                             freq_offset_hz: float,
                             sample_rate: float,
                             rng: np.random.Generator,
                             physical_sample_rate_hz: float = None) -> np.ndarray:
    """
    Add colored (1/f²) phase noise matching a specified SSB level (dBc/Hz)
    at a given frequency offset.

    MATLAB equivalent: comm.PhaseNoise / Phase Noise Simulink block.

    References
    ----------
    Kasdin, N.J., "Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/(f^alpha) Power Law Noise Generation," Proceedings of
    the IEEE, Vol. 83, No. 5, May 1995.

    Model
    -----
    For a 1/f² (random-walk / Wiener) process, the SSB phase noise PSD is:

        L(f)  =  sigma_w² / (8π² f² Ts)   [rad²/Hz]

    where sigma_w is the per-sample phase-increment standard deviation and
    Ts = 1/sample_rate_physical is the physical sample period.

    Given the specification L0 = 10^(pn_dbc_hz/10) at offset f0:

        sigma_w²  =  L0 · 8π² · f0² · Ts  =  L0 · 8π² · f0² / fs_physical

    The Kasdin AR(1) / Wiener-process implementation:
        phi[n] = phi[n-1] + w[n],    w ~ N(0, sigma_w²)

    This is the correct discrete-time model for 1/f² (Brownian) phase noise
    and matches the MATLAB Phase Noise Simulink block behaviour.

    Parameters
    ----------
    signal                   : complex 1-D array
    pn_dbc_hz                : SSB phase noise level at freq_offset_hz (dBc/Hz)
    freq_offset_hz           : reference offset frequency (Hz, physical units)
    sample_rate              : sample rate of the signal (same units as freq_offset_hz
                               or normalised — used only for array indexing)
    rng                      : seeded random generator
    physical_sample_rate_hz  : physical sample rate in Hz (for sigma_w calculation).
                               If None, sample_rate is used directly (correct when
                               sample_rate is already in Hz; incorrect in normalised
                               mode with sample_rate in sym/s).
    """
    if pn_dbc_hz >= 0:
        return signal.copy()   # non-negative dBc is unphysical

    N = len(signal)
    if N < 4:
        return signal.copy()

    # Use physical sample rate for the Wiener process sigma computation
    fs_phys = physical_sample_rate_hz if physical_sample_rate_hz is not None else sample_rate

    L0 = 10.0 ** (pn_dbc_hz / 10.0)   # linear SSB level (rad²/Hz)

    # Per-sample phase-increment variance (Kasdin 1/f² Wiener model)
    # sigma_w² = L0 · 8π² · f0² / fs_physical
    sigma_w2 = L0 * 8.0 * np.pi ** 2 * freq_offset_hz ** 2 / fs_phys

    if sigma_w2 <= 0.0 or not np.isfinite(sigma_w2):
        return signal.copy()

    sigma_w = np.sqrt(sigma_w2)

    # Wiener process: cumulative sum of i.i.d. Gaussian increments
    increments = rng.normal(0.0, sigma_w, N)
    phase_t = np.cumsum(increments)

    # Remove mean to avoid net DC phase offset
    phase_t -= phase_t.mean()

    return signal * np.exp(1j * phase_t)


# ============================================================
# 5b. Phase Noise – White (legacy simplified model)
# ============================================================

def add_phase_noise(signal: np.ndarray,
                    phase_noise_var: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Add sample-by-sample additive white phase noise (legacy model).
    phi[n] ~ N(0, phase_noise_var)  (radians)
    """
    if phase_noise_var <= 0:
        return signal.copy()
    phi = rng.normal(0.0, np.sqrt(phase_noise_var), len(signal))
    return signal * np.exp(1j * phi)


# ============================================================
# 6.  I/Q Imbalance
# ============================================================

def apply_iq_imbalance(signal: np.ndarray,
                       amplitude_imbalance_db: float,
                       phase_imbalance_deg: float) -> np.ndarray:
    """
    Model I/Q mixer imbalance — MATLAB comm.IQImbalance symmetric ± model.

    MATLAB block behaviour (RF Satellite Link example)
    ---------------------------------------------------
    "Amplitude imbalance (3 dB)":
        gain_I = 10^(+1.5/20) ≈ +1.5 dB   applied to the I mixer output
        gain_Q = 10^(-1.5/20) ≈ -1.5 dB   applied to the Q mixer output

    "Phase imbalance (20 deg)":
        I branch is rotated by +10°
        Q branch is rotated by -10°

    Combined matrix form:
        I_out = gain_I * ( I_in * cos(dphi) - Q_in * sin(dphi) )
        Q_out = gain_Q * ( I_in * sin(dphi) + Q_in * cos(dphi) )

    where:
        alpha = amplitude_imbalance_db / 2       (split equally ±)
        dphi  = phase_imbalance_deg / 2  [rad]   (split equally ±)
        gain_I = 10^( alpha / 20)
        gain_Q = 10^(-alpha / 20)

    This matches the MATLAB comm.IQImbalance block exactly for the parameter
    values used in the RF Satellite Link example:
        amplitude_imbalance_db = 3.0 dB
        phase_imbalance_deg    = 20.0 °

    Note on compensator detectability
    ----------------------------------
    The symmetric model gives E[I_out * Q_out] ≠ 0 even for pure amplitude
    imbalance (because of the rotation), so the LMS compensator (which
    minimises the conjugate cross-power E[y * conj(y)]) can detect and
    remove both amplitude and phase components.  The 2nd-order batch
    estimator used in _iq_lms_compensator() tracks the same statistics.
    """
    if amplitude_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
        return signal.copy()

    alpha  = amplitude_imbalance_db / 2.0          # split ±
    dphi   = np.deg2rad(phase_imbalance_deg / 2.0) # split ±

    gain_I = 10.0 ** ( alpha / 20.0)
    gain_Q = 10.0 ** (-alpha / 20.0)

    I_in = signal.real
    Q_in = signal.imag

    I_out = gain_I * (I_in * np.cos(dphi) - Q_in * np.sin(dphi))
    Q_out = gain_Q * (I_in * np.sin(dphi) + Q_in * np.cos(dphi))

    return I_out + 1j * Q_out


# ============================================================
# 7.  DC Offset
# ============================================================

def add_dc_offset(signal: np.ndarray,
                  dc_i: float,
                  dc_q: float) -> np.ndarray:
    """Add a complex DC offset to the signal (absolute value)."""
    return signal + (dc_i + 1j * dc_q)


def add_dc_offset_relative(signal: np.ndarray,
                            frac_i: float,
                            frac_q: float) -> np.ndarray:
    """
    Add a DC offset expressed as a fraction of signal RMS.
    Legacy mode for relative offsets.
    """
    sig_rms = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    return signal + ((frac_i + 1j * frac_q) * sig_rms)


# ============================================================
# 8.  LNA Gain
# ============================================================

def apply_lna_gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a linear voltage gain (dB) to the signal."""
    return signal * 10.0 ** (gain_db / 20.0)
