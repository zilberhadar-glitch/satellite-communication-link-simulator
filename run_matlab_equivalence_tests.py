"""
run_matlab_equivalence_tests.py
--------------------------------
MATLAB-equivalence test suite for the RF Satellite Link Python simulation.

Runs scenarios that directly correspond to the MATLAB/Simulink RF Satellite
Link example (https://www.mathworks.com/help/comm/ug/rf-satellite-link.html).

Results saved to:
    matlab_equivalence_results.csv
    matlab_equivalence_log.txt

Usage (PowerShell / CMD / bash):
    python run_matlab_equivalence_tests.py
"""

import copy
import csv
import sys
import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Ensure we pick up the updated modules from outputs/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import Config
from transmitter import transmit
from channel import propagate
from receiver import receive, attach_srrc_h
from metrics import ScenarioResult, compute_ebn0_db, compute_evm, compute_papr_db
from modulation import bits_to_symbols

OUT_DIR = "matlab_equivalence_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "matlab_equivalence_log.txt")
CSV_PATH = os.path.join(OUT_DIR, "matlab_equivalence_results.csv")


# ---------------------------------------------------------------------------
# Helper: run one scenario
# ---------------------------------------------------------------------------

def run_one(cfg: Config, name: str,
            override_doppler_hz: float = None,
            override_noise_temp_k: float = None,
            custom_backoff_db: float = None) -> ScenarioResult:
    attach_srrc_h(cfg)
    rng = np.random.default_rng(cfg.random_seed)

    tx = transmit(cfg, rng, custom_backoff_db=custom_backoff_db)
    ch = propagate(tx.after_hpa, cfg, rng,
                   override_doppler_hz=override_doppler_hz,
                   override_noise_temp_k=override_noise_temp_k)

    dop_for_rx = override_doppler_hz if (cfg.apply_doppler_correction
                                          and cfg.cfo_correction_mode == "ideal") else None
    rx = receive(ch.signal, tx.bits, cfg, override_doppler_hz=dop_for_rx)

    r = ScenarioResult(name)
    r.ber     = rx.ber
    r.ser     = rx.ser
    r.n_errors = rx.n_bit_errors
    r.snr_db  = ch.snr_db
    nt = override_noise_temp_k if override_noise_temp_k is not None else cfg.noise_temp_k
    r.ebn0_db = compute_ebn0_db(cfg, noise_temp_k=nt)
    tx_ideal  = bits_to_symbols(tx.bits, cfg.modulation_order)
    r.evm_pct = compute_evm(tx_ideal, rx.symbols)
    r.papr_db = compute_papr_db(tx.after_hpa)
    r.notes   = (f"DPD={'Y' if cfg.apply_dpd else 'N'} | "
                 f"HPA={'Y' if cfg.apply_hpa else 'bypass'} | "
                 f"CFO_mode={cfg.cfo_correction_mode}")
    return r


# ---------------------------------------------------------------------------
# Base configuration – MATLAB nominal
# ---------------------------------------------------------------------------

BASE = Config(
    modulation_order=16,
    carrier_freq_hz=4.0e9,
    sat_altitude_km=35_600.0,
    tx_antenna_diameter_m=0.4,
    rx_antenna_diameter_m=0.4,
    lna_gain_db=30.0,
    noise_temp_k=20.0,          # MATLAB default
    hpa_input_backoff_db=30.0,  # near-linear baseline
    apply_hpa=True,
    apply_dpd=False,
    doppler_hz=0.0,
    apply_doppler_correction=False,
    cfo_correction_mode="blind",
    rolloff=0.25,
    span=10,
    samples_per_symbol=8,
    num_symbols=10_000,
    random_seed=42,
    apply_phase_noise=False,
    apply_iq_imbalance=False,
    apply_dc_offset=False,
    apply_agc=True,
    verbose=False,
)


# ---------------------------------------------------------------------------
# Scenario groups
# ---------------------------------------------------------------------------

scenarios: list = []


# ── Group 1: MATLAB nominal ──────────────────────────────────────────────
print("Group 1: MATLAB nominal case")

cfg = copy.deepcopy(BASE)
scenarios.append(run_one(cfg, "G1-01 Nominal (20K, IBO=30, no imps)"))


# ── Group 2: HPA nonlinearity ────────────────────────────────────────────
print("Group 2: HPA nonlinearity")

for ibo, tag in [(30, "IBO=30"), (7, "IBO=7"), (1, "IBO=1")]:
    cfg = copy.deepcopy(BASE)
    cfg.apply_hpa = True
    cfg.apply_dpd = False
    scenarios.append(run_one(cfg, f"G2-HPA {tag} no DPD",
                             custom_backoff_db=ibo))

# HPA bypass
cfg = copy.deepcopy(BASE)
cfg.apply_hpa = False
scenarios.append(run_one(cfg, "G2-HPA bypass (ideal linear)"))

# IBO=7 with DPD
cfg = copy.deepcopy(BASE)
cfg.apply_hpa = True
cfg.apply_dpd = True
scenarios.append(run_one(cfg, "G2-HPA IBO=7 WITH DPD", custom_backoff_db=7))

# IBO=1 with DPD
cfg = copy.deepcopy(BASE)
cfg.apply_hpa = True
cfg.apply_dpd = True
scenarios.append(run_one(cfg, "G2-HPA IBO=1 WITH DPD", custom_backoff_db=1))


# ── Group 3: Noise temperature ───────────────────────────────────────────
print("Group 3: Noise temperature")

for T in [0, 20, 290, 500]:
    cfg = copy.deepcopy(BASE)
    scenarios.append(run_one(cfg, f"G3-Noise T={T}K",
                             override_noise_temp_k=float(T)))


# ── Group 4: Doppler / Carrier synchronisation ───────────────────────────
print("Group 4: Doppler / carrier sync")

# 0 Hz – baseline
cfg = copy.deepcopy(BASE)
scenarios.append(run_one(cfg, "G4-Doppler 0Hz (no sync needed)",
                         override_doppler_hz=0.0, override_noise_temp_k=20.0))

# 3 Hz – no correction
cfg = copy.deepcopy(BASE)
cfg.apply_doppler_correction = False
scenarios.append(run_one(cfg, "G4-Doppler 3Hz NO correction",
                         override_doppler_hz=3.0, override_noise_temp_k=20.0))

# 3 Hz – ideal correction
cfg = copy.deepcopy(BASE)
cfg.apply_doppler_correction = True
cfg.cfo_correction_mode = "ideal"
scenarios.append(run_one(cfg, "G4-Doppler 3Hz ideal correction",
                         override_doppler_hz=3.0, override_noise_temp_k=20.0))

# 3 Hz – blind batch estimator
cfg = copy.deepcopy(BASE)
cfg.apply_doppler_correction = True
cfg.cfo_correction_mode = "blind"
scenarios.append(run_one(cfg, "G4-Doppler 3Hz blind NDA estimator",
                         override_doppler_hz=3.0, override_noise_temp_k=20.0))

# 3 Hz – carrier_sync PLL (MATLAB comm.CarrierSynchronizer)
cfg = copy.deepcopy(BASE)
cfg.apply_doppler_correction = True
cfg.cfo_correction_mode = "carrier_sync"
cfg.carrier_sync_loop_bw = 0.01
cfg.carrier_sync_damping = 0.707
scenarios.append(run_one(cfg, "G4-Doppler 3Hz carrier_sync PLL",
                         override_doppler_hz=3.0, override_noise_temp_k=20.0))


# ── Group 5: Phase noise ─────────────────────────────────────────────────
print("Group 5: Phase noise")

# MATLAB default / negligible
cfg = copy.deepcopy(BASE)
cfg.apply_phase_noise = True
cfg.phase_noise_dbc_hz = -100.0
cfg.phase_noise_freq_offset_hz = 100.0
scenarios.append(run_one(cfg, "G5-Phase noise negligible (-100 dBc/Hz)"))

# Moderate
cfg = copy.deepcopy(BASE)
cfg.apply_phase_noise = True
cfg.phase_noise_dbc_hz = -85.0
cfg.phase_noise_freq_offset_hz = 100.0
scenarios.append(run_one(cfg, "G5-Phase noise moderate (-85 dBc/Hz)"))

# Severe
cfg = copy.deepcopy(BASE)
cfg.apply_phase_noise = True
cfg.phase_noise_dbc_hz = -60.0
cfg.phase_noise_freq_offset_hz = 100.0
scenarios.append(run_one(cfg, "G5-Phase noise severe (-60 dBc/Hz)"))

# Legacy white noise for comparison
cfg = copy.deepcopy(BASE)
cfg.apply_phase_noise = True
cfg.phase_noise_use_white = True
cfg.phase_noise_power_rad2 = 1e-3
scenarios.append(run_one(cfg, "G5-Phase noise white (1e-3 rad^2 legacy)"))


# ── Group 6: I/Q imbalance ───────────────────────────────────────────────
print("Group 6: I/Q imbalance")

# Amplitude-only (MATLAB: 3 dB)
cfg = copy.deepcopy(BASE)
cfg.apply_iq_imbalance = True
cfg.iq_amplitude_imbalance_db = 3.0
cfg.iq_phase_imbalance_deg = 0.0
cfg.apply_iq_correction = False
scenarios.append(run_one(cfg, "G6-IQ amp-only 3dB NO correction"))

cfg = copy.deepcopy(BASE)
cfg.apply_iq_imbalance = True
cfg.iq_amplitude_imbalance_db = 3.0
cfg.iq_phase_imbalance_deg = 0.0
cfg.apply_iq_correction = True
scenarios.append(run_one(cfg, "G6-IQ amp-only 3dB WITH correction"))

# Phase-only (MATLAB: 20 deg)
cfg = copy.deepcopy(BASE)
cfg.apply_iq_imbalance = True
cfg.iq_amplitude_imbalance_db = 0.0
cfg.iq_phase_imbalance_deg = 20.0
cfg.apply_iq_correction = False
scenarios.append(run_one(cfg, "G6-IQ phase-only 20deg NO correction"))

cfg = copy.deepcopy(BASE)
cfg.apply_iq_imbalance = True
cfg.iq_amplitude_imbalance_db = 0.0
cfg.iq_phase_imbalance_deg = 20.0
cfg.apply_iq_correction = True
scenarios.append(run_one(cfg, "G6-IQ phase-only 20deg WITH correction"))

# Combined (MATLAB default)
cfg = copy.deepcopy(BASE)
cfg.apply_iq_imbalance = True
cfg.iq_amplitude_imbalance_db = 3.0
cfg.iq_phase_imbalance_deg = 20.0
cfg.apply_iq_correction = False
scenarios.append(run_one(cfg, "G6-IQ combined 3dB+20deg NO correction"))

cfg = copy.deepcopy(BASE)
cfg.apply_iq_imbalance = True
cfg.iq_amplitude_imbalance_db = 3.0
cfg.iq_phase_imbalance_deg = 20.0
cfg.apply_iq_correction = True
scenarios.append(run_one(cfg, "G6-IQ combined 3dB+20deg WITH correction"))


# ── Group 7: DC offset ───────────────────────────────────────────────────
print("Group 7: DC offset")

# MATLAB absolute offsets, no correction
cfg = copy.deepcopy(BASE)
cfg.apply_dc_offset = True
cfg.dc_offset_mode = "absolute"
cfg.dc_offset_i_abs = 1e-8
cfg.dc_offset_q_abs = 5e-8
cfg.apply_dc_correction = False
scenarios.append(run_one(cfg, "G7-DC absolute (1e-8/5e-8) NO correction"))

# MATLAB absolute offsets, with correction
cfg = copy.deepcopy(BASE)
cfg.apply_dc_offset = True
cfg.dc_offset_mode = "absolute"
cfg.dc_offset_i_abs = 1e-8
cfg.dc_offset_q_abs = 5e-8
cfg.apply_dc_correction = True
scenarios.append(run_one(cfg, "G7-DC absolute (1e-8/5e-8) WITH correction"))

# Relative offsets (legacy)
cfg = copy.deepcopy(BASE)
cfg.apply_dc_offset = True
cfg.dc_offset_mode = "relative"
cfg.dc_offset_i = 0.02
cfg.dc_offset_q = 0.015
cfg.apply_dc_correction = True
scenarios.append(run_one(cfg, "G7-DC relative (2%/1.5%) WITH correction"))


# ---------------------------------------------------------------------------
# Print and save results
# ---------------------------------------------------------------------------

HDR = f"{'Scenario':<52} {'BER':>9} {'Eb/N0':>8} {'SNR':>8} {'EVM%':>7} {'PAPR':>7}"
SEP = "-" * len(HDR)

lines = [
    "=" * len(HDR),
    "RF Satellite Link – MATLAB Equivalence Test Results",
    "=" * len(HDR),
    HDR,
    SEP,
]
for s in scenarios:
    lines.append(
        f"  {s.name:<50} {s.ber:>9.3e} {s.ebn0_db:>8.1f} {s.snr_db:>8.1f} "
        f"{s.evm_pct:>7.1f} {s.papr_db:>7.1f}"
    )
lines.append("=" * len(HDR))

report = "\n".join(lines)
print("\n" + report)

with open(LOG_PATH, "w") as f:
    f.write(report + "\n\nNotes per scenario:\n")
    for s in scenarios:
        f.write(f"  {s.name}: {s.notes}\n")

with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Scenario", "BER", "SER", "N_errors",
                "EbN0_dB", "SNR_dB", "EVM_pct", "PAPR_dB", "Notes"])
    for s in scenarios:
        w.writerow([s.name, f"{s.ber:.6e}", f"{s.ser:.6e}", s.n_errors,
                    f"{s.ebn0_db:.2f}", f"{s.snr_db:.2f}",
                    f"{s.evm_pct:.2f}", f"{s.papr_db:.2f}", s.notes])

print(f"\nResults saved to:\n  {LOG_PATH}\n  {CSV_PATH}")
