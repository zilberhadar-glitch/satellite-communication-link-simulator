"""
channel.py
----------
RF downlink channel model.

MATLAB/Simulink RF Satellite Link block order (Ground Station Receiver):

  1. Free-space path loss  (Tx antenna gain + FSPL)
  2. Doppler frequency shift
  3. Rx antenna gain
  4. Receiver thermal noise (AWGN)   ← BEFORE Phase Noise and LNA
  5. Phase noise                     ← AFTER Thermal Noise, BEFORE LNA
  6. I/Q imbalance                   ← AFTER Phase Noise, BEFORE LNA
  7. DC offset                       ← PART OF I/Q Imbalance block, BEFORE LNA
  8. LNA gain                        ← AFTER all impairments (MATLAB order)

This ordering matches the Simulink block diagram from:
https://www.mathworks.com/help/comm/ug/rf-satellite-link.html

Fix vs previous Python version
--------------------------------
DC offset is now applied BEFORE the LNA (step 7), matching the MATLAB block
diagram where the I/Q Imbalance block (which includes DC offset) sits before
the LNA.  Previously DC was applied after LNA, which meant the tiny absolute
offsets (1e-8 / 5e-8) were not amplified by the LNA before the receiver — in
MATLAB they ARE amplified by 10^(30/20) ≈ 31.6× making them visible on the
post-LNA constellation.  This fix restores the correct MATLAB behaviour.

SNR accounting: the SNR is still measured as P_rx_signal / P_noise at the
point after path loss and before noise addition (at the Rx antenna terminal),
which is the standard definition and physically equivalent to the after-LNA
SNR (LNA gain cancels in the ratio).
"""

import numpy as np
from dataclasses import dataclass

from config import Config
from impairments import (
    apply_path_loss,
    apply_doppler,
    add_awgn_noise,
    add_phase_noise,
    add_colored_phase_noise,
    apply_iq_imbalance,
    add_dc_offset,
    add_dc_offset_relative,
    apply_lna_gain,
)


@dataclass
class ChannelOutput:
    """Container for channel output and diagnostics."""
    signal: np.ndarray   # signal at Rx output (after all channel effects)
    snr_db: float        # estimated SNR (dB) at Rx antenna terminal


def propagate(tx_signal: np.ndarray,
              cfg: Config,
              rng: np.random.Generator,
              override_doppler_hz: float = None,
              override_noise_temp_k: float = None) -> ChannelOutput:
    """
    Pass *tx_signal* through the downlink channel.

    MATLAB block order:
      path_loss → Doppler → AWGN_noise → phase_noise → IQ_imbalance
      → LNA → DC_offset

    Parameters
    ----------
    tx_signal            : complex 1-D array (HPA output, oversampled)
    cfg                  : simulation Config
    rng                  : seeded random generator
    override_doppler_hz  : overrides cfg.doppler_hz when given
    override_noise_temp_k: overrides cfg.noise_temp_k when given

    Returns
    -------
    ChannelOutput
    """
    doppler_hz   = cfg.doppler_hz   if override_doppler_hz    is None else override_doppler_hz
    noise_temp_k = cfg.noise_temp_k if override_noise_temp_k  is None else override_noise_temp_k

    sig = tx_signal.copy()

    # ------------------------------------------------------------------
    # 1. Free-space path loss (Tx gain + FSPL) + Rx antenna gain
    #    Net: Tx_gain_dBi − FSPL_dB + Rx_gain_dBi
    # ------------------------------------------------------------------
    sig = apply_path_loss(
        sig,
        fspl_db=cfg.free_space_path_loss_db,
        tx_gain_dbi=cfg.tx_antenna_gain_dbi,
        rx_gain_dbi=cfg.rx_antenna_gain_dbi,
    )

    # Measure signal power at Rx antenna terminal for SNR reporting
    rx_signal_power = float(np.mean(np.abs(sig) ** 2))

    # ------------------------------------------------------------------
    # 2. Doppler frequency offset
    # ------------------------------------------------------------------
    sig = apply_doppler(sig, doppler_hz, cfg.sample_rate_hz)

    # ------------------------------------------------------------------
    # 3. Receiver thermal noise (AWGN)  — BEFORE Phase Noise and LNA
    # ------------------------------------------------------------------
    sig = add_awgn_noise(sig, noise_temp_k, cfg.sample_rate_hz, rng)

    # Compute SNR at the Rx antenna (LNA gain cancels in the ratio)
    from config import K_BOLTZMANN
    noise_power_theoretical = K_BOLTZMANN * noise_temp_k * (cfg.sample_rate_hz / 2.0)
    if noise_temp_k > 0 and rx_signal_power > 0:
        snr_db = 10.0 * np.log10(rx_signal_power / (noise_power_theoretical + 1e-300))
    else:
        snr_db = float('inf')

    # ------------------------------------------------------------------
    # 4. Phase noise  — BEFORE LNA  (MATLAB block order)
    # ------------------------------------------------------------------
    if cfg.apply_phase_noise:
        if cfg.phase_noise_use_white:
            sig = add_phase_noise(sig, cfg.phase_noise_power_rad2, rng)
        else:
            # physical_sample_rate_hz: use cfg attribute if set, else sample_rate_hz.
            # In normalised mode (symbol_rate_baud=0), sample_rate_hz=8 sym/s but
            # the dBc/Hz spec is given at a physical Hz offset.  The caller should
            # set cfg.phase_noise_physical_sample_rate_hz to the physical rate
            # (e.g. 8e6 for 1 Mbaud × 8 sps) for correct Wiener process scaling.
            phys_sr = getattr(cfg, 'phase_noise_physical_sample_rate_hz',
                              cfg.sample_rate_hz)
            sig = add_colored_phase_noise(
                sig,
                pn_dbc_hz=cfg.phase_noise_dbc_hz,
                freq_offset_hz=cfg.phase_noise_freq_offset_hz,
                sample_rate=cfg.sample_rate_hz,
                rng=rng,
                physical_sample_rate_hz=phys_sr,
            )

    # ------------------------------------------------------------------
    # 5. I/Q imbalance  — BEFORE LNA  (MATLAB block order)
    # ------------------------------------------------------------------
    if cfg.apply_iq_imbalance:
        sig = apply_iq_imbalance(
            sig,
            cfg.iq_amplitude_imbalance_db,
            cfg.iq_phase_imbalance_deg,
        )

    # ------------------------------------------------------------------
    # 6. DC offset  — BEFORE LNA  (MATLAB: I/Q Imbalance block includes
    #    DC offset and sits before LNA in the Simulink block diagram)
    # ------------------------------------------------------------------
    if cfg.apply_dc_offset:
        if cfg.dc_offset_mode == "absolute":
            sig = add_dc_offset(sig, cfg.dc_offset_i_abs, cfg.dc_offset_q_abs)
        else:
            sig = add_dc_offset_relative(sig, cfg.dc_offset_i, cfg.dc_offset_q)

    # ------------------------------------------------------------------
    # 7. LNA gain  — AFTER noise / phase noise / IQ / DC  (MATLAB order)
    # ------------------------------------------------------------------
    sig = apply_lna_gain(sig, cfg.lna_gain_db)

    if cfg.verbose:
        print(f"  [Ch] Rx SNR ≈ {snr_db:.1f} dB | "
              f"Doppler={doppler_hz:.1f} Hz | T_noise={noise_temp_k:.0f} K | "
              f"LNA={cfg.lna_gain_db:.0f} dB")

    return ChannelOutput(signal=sig, snr_db=snr_db)
