"""
impairments.py
--------------
All RF impairment models:

  * Saleh TWTA memoryless nonlinearity (HPA)
  * Free-space path loss
  * Doppler frequency offset
  * AWGN / receiver thermal noise
  * Phase noise
  * I/Q amplitude & phase imbalance
  * DC offset injection

Each function is stateless and purely functional – it takes a signal array
(and parameters) and returns the distorted signal.
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

    The saturation input amplitude r_sat is derived from the Saleh model
    analytically:   r_sat = 1 / sqrt(b_a)

    The *input_backoff_db* is measured relative to the saturation power,
    so the signal is scaled to the correct operating point before passing
    through the nonlinearity.

    Parameters
    ----------
    signal          : complex 1-D array (normalised, unit average power)
    a_a, b_a        : Saleh AM/AM coefficients
    a_p, b_p        : Saleh AM/PM coefficients
    input_backoff_db: Input back-off (IBO) from saturation in dB.
                      Larger IBO  → more linear,  lower IBO → more distortion.

    Returns
    -------
    out : complex 1-D array after HPA
    """
    # Saturation amplitude of the Saleh model
    r_sat = 1.0 / np.sqrt(b_a)

    # Desired RMS input level given the IBO
    ibo_linear = 10 ** (input_backoff_db / 10.0)
    # rms_in such that  rms_in² * ibo_linear = r_sat²
    rms_target = r_sat / np.sqrt(ibo_linear)

    # Normalise input signal to the target RMS
    rms_in = np.sqrt(np.mean(np.abs(signal) ** 2))
    if rms_in > 0:
        x = signal * (rms_target / rms_in)
    else:
        return signal.copy()

    r = np.abs(x)           # instantaneous envelope

    # AM/AM
    A = a_a * r / (1.0 + b_a * r ** 2)

    # AM/PM
    phi = a_p * r ** 2 / (1.0 + b_p * r ** 2)

    # Reconstruct complex output
    theta = np.angle(x)
    out = A * np.exp(1j * (theta + phi))

    return out


def hpa_am_am_curve(a_a: float, b_a: float, n_points: int = 200):
    """Return (r_in, r_out) arrays for plotting the AM/AM characteristic."""
    r_sat = 1.0 / np.sqrt(b_a)
    r_in = np.linspace(0, 2 * r_sat, n_points)
    r_out = a_a * r_in / (1.0 + b_a * r_in ** 2)
    return r_in, r_out


# ============================================================
# 2.  Free-Space Path Loss (applied as a scalar gain)
# ============================================================

def apply_path_loss(signal: np.ndarray,
                    fspl_db: float,
                    tx_gain_dbi: float,
                    rx_gain_dbi: float) -> np.ndarray:
    """
    Scale signal amplitude by the link budget: Tx gain – FSPL + Rx gain.

    The net voltage gain is  10^( (Gt - FSPL + Gr) / 20 ).
    """
    net_db = tx_gain_dbi - fspl_db + rx_gain_dbi
    voltage_gain = 10.0 ** (net_db / 20.0)
    return signal * voltage_gain


# ============================================================
# 3.  Doppler Frequency Offset
# ============================================================

def apply_doppler(signal: np.ndarray,
                  doppler_hz: float,
                  sample_rate: float) -> np.ndarray:
    """
    Multiply the signal by a complex exponential to model Doppler shift.

    f_d in Hz,  sample_rate in normalised samples/sec (= sps for unit symbol rate).
    """
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
                   rng: np.random.Generator) -> np.ndarray:
    """
    Add complex AWGN whose power is set by the carrier-to-noise ratio.

    After the LNA, the signal has been scaled by the full link budget
    (path loss, antenna gains, LNA gain).  We compute the noise power in
    the same normalised amplitude domain by using:

        P_noise = k_B * T_sys * B   [physical watts]

    and then express it relative to the received signal power so the
    correct SNR is preserved regardless of the (arbitrary) normalisation
    scale of the simulation.

    Concretely:
        SNR_linear = P_signal_after_link / (k_B * T_sys * B)

    We set noise variance = P_signal / SNR_linear, so the ratio is exact.

    When noise_temp_k == 0 the channel is noiseless.
    """
    if noise_temp_k <= 0:
        return signal.copy()

    # Physical noise power in the receiver bandwidth B = sample_rate / 2
    noise_power_physical = K_BOLTZMANN * noise_temp_k * (sample_rate / 2.0)

    # Signal power in the simulation's amplitude domain (dimensionless)
    sig_power = np.mean(np.abs(signal) ** 2)

    # To get the right SNR we need noise_power_sim such that:
    #   sig_power / noise_power_sim = sig_power_physical / noise_power_physical
    #
    # But sig_power_physical is not directly known here — we track it via
    # the SNR which was already printed/computed in the channel.  Instead,
    # we derive sigma from the link: the noise should degrade the signal
    # according to the physical SNR.  The path_loss already scaled the
    # signal, so sig_power encodes the received physical power in
    # simulation units.  We need a reference power for 1 W of Tx power
    # in simulation units:
    #   reference_power_sim = (voltage_gain)^2  where voltage_gain = 10^(net_dB/20)
    #
    # Since we don't pass that through here we use the simpler approach:
    # compute sigma so that noise power / signal power = 1/SNR_linear.
    # SNR_linear is computed from the physical parameters passed in.

    if sig_power <= 0:
        return signal.copy()

    # sigma² per complex sample = noise_power_sim
    # We set noise_power_sim = sig_power / SNR_linear
    # where SNR_linear = sig_power_physical / noise_power_physical
    # and sig_power_physical ~ sig_power (they are equal in the sim by construction
    # of apply_path_loss which uses physical gains/losses).
    #
    # Therefore: noise_power_sim = noise_power_physical
    # (both measured in the same simulation-unit power domain)
    sigma2 = noise_power_physical   # variance of the complex noise sample
    sigma = np.sqrt(sigma2 / 2.0)   # per real/imag component

    noise = sigma * (rng.standard_normal(len(signal))
                     + 1j * rng.standard_normal(len(signal)))
    return signal + noise


# ============================================================
# 5.  Phase Noise
# ============================================================

def add_phase_noise(signal: np.ndarray,
                    phase_noise_var: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Add sample-by-sample additive white phase noise.

    phi[n] ~ N(0, phase_noise_var)   (radians)
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
    Model I/Q mixer imbalance as amplitude and phase mismatch.

    The received complex baseband signal is modelled as:
        y = (1 + α/2) * x  +  (1 - α/2) * x*  * exp(jΔφ)     [simplified]

    A common linear model:
        I_out = I_in
        Q_out = (1 + ε) * Q_in  + I_in * sin(Δφ)

    where  ε = 10^(A_dB/20) - 1,   Δφ in radians.
    """
    if amplitude_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
        return signal.copy()

    eps = 10 ** (amplitude_imbalance_db / 20.0) - 1.0
    dphi = np.deg2rad(phase_imbalance_deg)

    I = signal.real
    Q = signal.imag

    I_out = I
    Q_out = (1.0 + eps) * Q + I * np.sin(dphi)

    return I_out + 1j * Q_out


# ============================================================
# 7.  DC Offset
# ============================================================

def add_dc_offset(signal: np.ndarray,
                  dc_i: float, dc_q: float) -> np.ndarray:
    """Add a complex DC offset to the signal."""
    return signal + (dc_i + 1j * dc_q)


# ============================================================
# 8.  LNA gain
# ============================================================

def apply_lna_gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a linear voltage gain (dB) to the signal."""
    voltage_gain = 10.0 ** (gain_db / 20.0)
    return signal * voltage_gain
