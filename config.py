"""
config.py
---------
Centralised configuration for the RF Satellite Link simulation.
Updated to match the MATLAB/Simulink RF Satellite Link example more closely.

Key changes vs original Python:
  - Default noise_temp_k changed from 290 K → 20 K  (matches MATLAB nominal)
  - Added apply_hpa / symbol_rate_baud (physical units)
  - Added DPD parameters (apply_dpd, dpd_lut_points)
  - Added colored phase-noise parameters (phase_noise_dbc_hz, phase_noise_freq_offset_hz)
  - Added dc_offset_mode ("absolute" | "relative") + absolute offset fields
  - Added cfo_correction_mode ("blind" | "ideal" | "carrier_sync")
  - carrier_sync loop-bandwidth parameter
  - Separate I/Q amplitude-only / phase-only test fields kept as scenario overrides
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT     = 2.998e8        # Speed of light (m/s)
K_BOLTZMANN = 1.3806e-23     # Boltzmann constant (J/K)
EARTH_RADIUS_KM = 6371.0     # Earth mean radius (km)


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Modulation
    # ------------------------------------------------------------------
    modulation_order:  int   = 16       # QAM order  (must be 4, 16, 64, …)
    bits_per_symbol:   int   = 4        # log2(16) – recomputed in update()

    # ------------------------------------------------------------------
    # Pulse shaping filter
    # ------------------------------------------------------------------
    rolloff:           float = 0.25     # Excess bandwidth factor β
    span:              int   = 10       # Filter span in symbols
    samples_per_symbol: int  = 8        # Upsampling factor

    # ------------------------------------------------------------------
    # Link budget
    # ------------------------------------------------------------------
    carrier_freq_hz:          float = 4.0e9      # Downlink carrier (Hz)
    sat_altitude_km:          float = 35_600.0   # GEO altitude (km)
    tx_antenna_diameter_m:    float = 0.4
    rx_antenna_diameter_m:    float = 0.4
    antenna_efficiency:       float = 0.55       # Aperture efficiency

    # Physical symbol rate (baud).  Set to a positive value to use physical
    # units for Doppler Hz, noise BW, etc.  Set to 0 to keep normalised mode
    # (symbol_rate = 1 sym/s, matching the original Python behaviour).
    symbol_rate_baud: float = 0.0    # 0 → normalised (legacy mode)

    # MATLAB's Doppler value is specified in physical Hz.  In normalised
    # mode we still need a physical reference symbol rate so that 3 Hz is
    # treated as a small carrier offset, not as 3 cycles/symbol.
    # This preserves the MATLAB experiment behaviour while keeping the
    # rest of the link in normalised units.
    doppler_reference_symbol_rate_baud: float = 1_000.0

    # ------------------------------------------------------------------
    # HPA (Saleh TWTA memoryless model)
    # ------------------------------------------------------------------
    hpa_saleh_a_a: float = 2.1587
    hpa_saleh_b_a: float = 1.1517
    hpa_saleh_a_p: float = 4.0033
    hpa_saleh_b_p: float = 9.1040
    hpa_input_backoff_db: float = 7.0
    apply_hpa: bool = True    # False → bypass (ideal linear amplifier)

    # ------------------------------------------------------------------
    # DPD – Digital Pre-Distortion  (NEW – MATLAB has this subsystem)
    # ------------------------------------------------------------------
    apply_dpd: bool = False
    dpd_lut_points: int = 1024   # Resolution of the inverse-Saleh LUT

    # ------------------------------------------------------------------
    # Doppler
    # ------------------------------------------------------------------
    doppler_hz: float = 3.0
    apply_doppler_correction: bool = True

    # CFO correction mode:
    #   "blind"            – 4th-power NDA batch estimator (original Python)
    #   "ideal"            – known true CFO fed in directly
    #   "carrier_sync"     – simplified 2nd-order PLL (closer to MATLAB
    #                        comm.CarrierSynchronizer)
    cfo_correction_mode: str = "blind"

    # PLL parameters (used only when cfo_correction_mode == "carrier_sync")
    carrier_sync_loop_bw: float = 0.01    # Normalised loop bandwidth  BL*Ts
    carrier_sync_damping: float = 0.707   # Damping factor ζ (Butterworth)

    # ------------------------------------------------------------------
    # Receiver thermal noise
    # ------------------------------------------------------------------
    # MATLAB default is 20 K; original Python used 290 K.
    noise_temp_k: float = 20.0     # System noise temperature (K) – MATLAB default
    lna_gain_db:  float = 30.0     # Low-noise amplifier gain (dB)

    # ------------------------------------------------------------------
    # Phase noise  (MATLAB uses colored / dBc-Hz model)
    # ------------------------------------------------------------------
    apply_phase_noise: bool = False

    # NEW: colored phase-noise model (matches MATLAB Phase Noise block)
    phase_noise_dbc_hz:         float = -85.0   # Level at offset (dBc/Hz)
    phase_noise_freq_offset_hz: float = 100.0   # Offset frequency (Hz, physical)

    # Physical sample rate used for Wiener sigma_w computation in normalised mode.
    # Set to the physical sample rate (Hz) matching freq_offset_hz units.
    # MATLAB RF Satellite Link: ~1 Mbaud x 8 sps = 8 MHz.
    # When symbol_rate_baud > 0, this is overridden by update().
    phase_noise_physical_sample_rate_hz: float = 8_000_000.0

    # LEGACY: white phase noise (kept for backward compatibility)
    phase_noise_use_white: bool = False
    phase_noise_power_rad2: float = 1e-4   # White noise variance (rad²)

    # ------------------------------------------------------------------
    # I/Q imbalance
    # ------------------------------------------------------------------
    apply_iq_imbalance:       bool  = False
    iq_amplitude_imbalance_db: float = 3.0    # MATLAB: 3 dB amplitude imbalance
    iq_phase_imbalance_deg:   float = 20.0   # MATLAB: 20° phase imbalance
    apply_iq_correction:      bool  = True

    # ------------------------------------------------------------------
    # DC offset
    # ------------------------------------------------------------------
    apply_dc_offset: bool = False

    # Mode: "absolute"  → values in the same units as the (normalised) signal
    #        "relative" → values as fractions of signal RMS (original Python)
    dc_offset_mode: str = "relative"

    # DC offset values.
    #
    # MATLAB specifies physical volt offsets: I=1e-8, Q=5e-8.
    # The main Python simulation uses normalised signal amplitudes, so the
    # MATLAB volt values cannot be inserted directly.  For the MATLAB-equivalent
    # normalised path we use fractions of signal RMS chosen to reproduce the
    # documented MATLAB behaviour:
    #   * I offset: shifts the constellation but does not create errors alone.
    #   * Q offset: creates errors and a visible DC component.
    # The raw absolute fields remain available only for a fully physical-power
    # extension, and channel.py warns if absolute mode is used with normalised
    # symbol rate.
    dc_offset_i_abs: float = 1e-8
    dc_offset_q_abs: float = 5e-8

    # MATLAB-equivalent normalised DC offsets, as fractions of signal RMS.
    dc_offset_i: float = 0.05   # approx. MATLAB I DC 1e-8 behaviour
    dc_offset_q: float = 0.18   # approx. MATLAB Q DC 5e-8 behaviour
    apply_dc_correction: bool = True

    # ------------------------------------------------------------------
    # AGC
    # ------------------------------------------------------------------
    apply_agc:         bool  = True
    agc_target_power:  float = 1.0

    # ------------------------------------------------------------------
    # Simulation size
    # ------------------------------------------------------------------
    num_symbols: int = 10_000

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    random_seed: int  = 42
    verbose:     bool = True

    # ------------------------------------------------------------------
    # Derived (recomputed by update())
    # ------------------------------------------------------------------
    symbol_rate_baud_eff: float = field(init=False)   # effective symbol rate
    sample_rate_hz:       float = field(init=False)
    doppler_symbol_rate_eff: float = field(init=False)
    doppler_sample_rate_hz:  float = field(init=False)
    wavelength_m:         float = field(init=False)
    free_space_path_loss_db: float = field(init=False)
    tx_antenna_gain_dbi:  float = field(init=False)
    rx_antenna_gain_dbi:  float = field(init=False)

    def __post_init__(self):
        self.update()

    def update(self):
        """Recompute derived quantities after any parameter change."""
        self.bits_per_symbol = int(math.log2(self.modulation_order))

        # Symbol / sample rates
        if self.symbol_rate_baud > 0:
            self.symbol_rate_baud_eff = float(self.symbol_rate_baud)
            # Physical sample rate overrides the default for phase noise scaling
            self.phase_noise_physical_sample_rate_hz = self.symbol_rate_baud_eff * self.samples_per_symbol
        else:
            self.symbol_rate_baud_eff = 1.0   # normalised
        self.sample_rate_hz = self.symbol_rate_baud_eff * self.samples_per_symbol

        # Separate Doppler timebase.  If the entire simulation is physical,
        # use the physical symbol rate.  If the main simulation is normalised,
        # use the MATLAB-reference symbol rate only for phase/frequency offset
        # conversion.  This fixes the MATLAB Doppler=3 Hz case without changing
        # the normalised link-budget/noise scaling.
        if self.symbol_rate_baud > 0:
            self.doppler_symbol_rate_eff = self.symbol_rate_baud_eff
        else:
            self.doppler_symbol_rate_eff = float(self.doppler_reference_symbol_rate_baud)
        self.doppler_sample_rate_hz = self.doppler_symbol_rate_eff * self.samples_per_symbol

        self.wavelength_m = C_LIGHT / self.carrier_freq_hz

        # Free-space path loss  FSPL = (4πd/λ)²
        dist_m = self.sat_altitude_km * 1e3
        self.free_space_path_loss_db = 20 * math.log10(
            4 * math.pi * dist_m / self.wavelength_m
        )

        # Dish antenna gain  G = η (π D / λ)²
        self.tx_antenna_gain_dbi = 10 * math.log10(
            self.antenna_efficiency
            * (math.pi * self.tx_antenna_diameter_m / self.wavelength_m) ** 2
        )
        self.rx_antenna_gain_dbi = 10 * math.log10(
            self.antenna_efficiency
            * (math.pi * self.rx_antenna_diameter_m / self.wavelength_m) ** 2
        )

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "RF Satellite Link – Configuration Summary",
            "=" * 65,
            f"  Modulation            : {self.modulation_order}-QAM",
            f"  Carrier frequency     : {self.carrier_freq_hz/1e9:.2f} GHz",
            f"  Satellite altitude    : {self.sat_altitude_km:.0f} km",
            f"  Free-space path loss  : {self.free_space_path_loss_db:.1f} dB",
            f"  Tx antenna gain       : {self.tx_antenna_gain_dbi:.1f} dBi",
            f"  Rx antenna gain       : {self.rx_antenna_gain_dbi:.1f} dBi",
            f"  LNA gain              : {self.lna_gain_db:.1f} dB",
            f"  Noise temperature     : {self.noise_temp_k:.0f} K",
            f"  HPA bypass            : {not self.apply_hpa}",
            f"  HPA input back-off    : {self.hpa_input_backoff_db:.1f} dB",
            f"  DPD enabled           : {self.apply_dpd}",
            f"  Doppler offset        : {self.doppler_hz:.1f} Hz",
            f"  Doppler ref. sym rate : {self.doppler_symbol_rate_eff:.0f} Baud",
            f"  CFO correction mode   : {self.cfo_correction_mode}",
            f"  Phase noise           : {self.apply_phase_noise}",
            f"  I/Q imbalance         : {self.apply_iq_imbalance}",
            f"    Amplitude imbalance : {self.iq_amplitude_imbalance_db:.1f} dB",
            f"    Phase imbalance     : {self.iq_phase_imbalance_deg:.1f} deg",
            f"  DC offset             : {self.apply_dc_offset} (mode={self.dc_offset_mode})",
            f"  Num symbols           : {self.num_symbols}",
            f"  Symbol rate           : {'normalised' if self.symbol_rate_baud == 0 else f'{self.symbol_rate_baud_eff:.0f} Baud'}",
            "=" * 65,
        ]
        return "\n".join(lines)
