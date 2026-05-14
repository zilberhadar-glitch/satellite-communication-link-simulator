"""
config.py
---------
Centralised configuration for the RF Satellite Link simulation.
All physical and simulation parameters live here so that every module
imports from a single source of truth.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT = 2.998e8        # Speed of light (m/s)
K_BOLTZMANN = 1.3806e-23  # Boltzmann constant (J/K)
EARTH_RADIUS_KM = 6371.0  # Earth mean radius (km)


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Modulation
    # ------------------------------------------------------------------
    modulation_order: int = 16        # QAM order  (must be 4, 16, 64, …)
    bits_per_symbol: int = 4          # log2(16)

    # ------------------------------------------------------------------
    # Pulse shaping filter
    # ------------------------------------------------------------------
    rolloff: float = 0.25             # Excess bandwidth factor
    span: int = 10                    # Filter span in symbols
    samples_per_symbol: int = 8       # Upsampling factor

    # ------------------------------------------------------------------
    # Link budget
    # ------------------------------------------------------------------
    carrier_freq_hz: float = 4.0e9    # Downlink carrier frequency (Hz)
    sat_altitude_km: float = 35_600.0 # Geostationary orbit altitude (km)
    tx_antenna_diameter_m: float = 0.4
    rx_antenna_diameter_m: float = 0.4
    antenna_efficiency: float = 0.55  # Typical aperture efficiency

    # ------------------------------------------------------------------
    # HPA (Saleh TWTA memoryless model)
    # ------------------------------------------------------------------
    # Saleh AM/AM coefficients  a_a, b_a  →  G_AM(r) = a_a*r / (1 + b_a*r²)
    hpa_saleh_a_a: float = 2.1587
    hpa_saleh_b_a: float = 1.1517
    # Saleh AM/PM coefficients  a_p, b_p  →  Φ(r) = a_p*r² / (1 + b_p*r²)  [rad]
    hpa_saleh_a_p: float = 4.0033
    hpa_saleh_b_p: float = 9.1040
    # Input back-off from saturation (dB).  Large → linear region; small → compressed
    hpa_input_backoff_db: float = 7.0  # default; overridden per scenario

    # ------------------------------------------------------------------
    # Doppler
    # ------------------------------------------------------------------
    doppler_hz: float = 3.0           # Simulated Doppler offset (Hz)
    apply_doppler_correction: bool = True

    # ------------------------------------------------------------------
    # Receiver thermal noise
    # ------------------------------------------------------------------
    noise_temp_k: float = 290.0       # System noise temperature (K)
    lna_gain_db: float = 30.0         # Low-noise amplifier gain (dB)

    # ------------------------------------------------------------------
    # Phase noise
    # ------------------------------------------------------------------
    apply_phase_noise: bool = False
    phase_noise_power_rad2: float = 1e-4  # Variance of additive phase noise

    # ------------------------------------------------------------------
    # I/Q imbalance
    # ------------------------------------------------------------------
    apply_iq_imbalance: bool = False
    iq_amplitude_imbalance_db: float = 1.0   # dB
    iq_phase_imbalance_deg: float = 5.0      # degrees
    apply_iq_correction: bool = True

    # ------------------------------------------------------------------
    # DC offset
    # ------------------------------------------------------------------
    apply_dc_offset: bool = False
    dc_offset_i: float = 0.02
    dc_offset_q: float = 0.015
    apply_dc_correction: bool = True

    # ------------------------------------------------------------------
    # AGC
    # ------------------------------------------------------------------
    apply_agc: bool = True
    agc_target_power: float = 1.0

    # ------------------------------------------------------------------
    # Simulation size
    # ------------------------------------------------------------------
    num_symbols: int = 10_000

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    random_seed: int = 42
    verbose: bool = True

    # ------------------------------------------------------------------
    # Derived (computed lazily – call .update() after changing params)
    # ------------------------------------------------------------------
    symbol_rate_baud: float = field(init=False)
    sample_rate_hz: float = field(init=False)
    wavelength_m: float = field(init=False)
    free_space_path_loss_db: float = field(init=False)
    tx_antenna_gain_dbi: float = field(init=False)
    rx_antenna_gain_dbi: float = field(init=False)

    def __post_init__(self):
        self.update()

    def update(self):
        """Recompute derived quantities after any parameter change."""
        import math
        self.bits_per_symbol = int(math.log2(self.modulation_order))
        self.symbol_rate_baud = 1.0                      # normalised
        self.sample_rate_hz = self.symbol_rate_baud * self.samples_per_symbol

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
            "=" * 60,
            "RF Satellite Link – Configuration Summary",
            "=" * 60,
            f"  Modulation          : {self.modulation_order}-QAM",
            f"  Carrier frequency   : {self.carrier_freq_hz/1e9:.2f} GHz",
            f"  Satellite altitude  : {self.sat_altitude_km:.0f} km",
            f"  Free-space path loss: {self.free_space_path_loss_db:.1f} dB",
            f"  Tx antenna gain     : {self.tx_antenna_gain_dbi:.1f} dBi",
            f"  Rx antenna gain     : {self.rx_antenna_gain_dbi:.1f} dBi",
            f"  LNA gain            : {self.lna_gain_db:.1f} dB",
            f"  Noise temperature   : {self.noise_temp_k:.0f} K",
            f"  HPA input back-off  : {self.hpa_input_backoff_db:.1f} dB",
            f"  Doppler offset      : {self.doppler_hz:.1f} Hz",
            f"  Phase noise         : {self.apply_phase_noise}",
            f"  I/Q imbalance       : {self.apply_iq_imbalance}",
            f"  DC offset           : {self.apply_dc_offset}",
            f"  Num symbols         : {self.num_symbols}",
            "=" * 60,
        ]
        return "\n".join(lines)
