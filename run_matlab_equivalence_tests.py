"""
run_matlab_equivalence_tests.py
--------------------------------
MATLAB-equivalence test suite for the RF Satellite Link Python simulation.

Reference:
    https://www.mathworks.com/help/comm/ug/rf-satellite-link.html

STRUCTURE
---------
Section A  — MATLAB-equivalent scenarios
    These directly reproduce the parameter settings described on the MathWorks
    page.  Each test maps to a specific MATLAB Model Parameters choice.
    Results should qualitatively match the behaviour described in the
    MATLAB documentation.

Section B  — Extra validation (non-MATLAB)
    Additional sweeps and ideal-correction tests that go beyond the MATLAB
    example.  These are clearly marked as "extra" and are not claimed to be
    MATLAB-equivalent.

FIDELITY NOTES (per block)
--------------------------
Exact matches:
    Bit source, 16-QAM, SRRC filters, Saleh HPA, FSPL, Doppler,
    thermal noise, IQ imbalance (symmetric ± model), LNA, QAM demod, BER.

Closest approximations (documented):
    Phase noise     — Wiener 1/f² model vs MATLAB's interpolated FIR table.
    DC offset       — normalised-equivalent fractions vs MATLAB physical volts
                      (see config.py for derivation).
    DC Blocker      — IIR y[n]=x[n]-x[n-1]+α·y[n-1] vs dsp.DCBlocker.
    AGC             — static batch normalisation vs comm.AGC adaptive loop
                      (equivalent for stationary channel).
    IQ compensator  — DD two-stage LMS vs comm.IQImbalanceCompensator.
    Carrier sync    — coarse 4th-power + DA phase + fine DD-PLL vs single
                      comm.CarrierSynchronizer.  In normalised mode, Doppler Hz
                      is converted through Config.doppler_reference_symbol_rate_baud
                      so the MATLAB 3 Hz scenario is physically meaningful.
    DPD             — analytic Saleh inverse LUT vs MATLAB's DPD subsystem.

OUTPUT
------
    matlab_equivalence_outputs/matlab_equivalence_results.csv
    matlab_equivalence_outputs/matlab_equivalence_log.txt

Usage:
    python run_matlab_equivalence_tests.py
"""

import copy, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from transmitter import transmit
from channel import propagate
from receiver import receive, attach_srrc_h, iq_correct_ideal
from filters import rx_filter, filter_delay
from metrics import ScenarioResult, compute_ebn0_db, compute_evm, compute_papr_db
from modulation import bits_to_symbols, symbols_to_bits, symbol_error_rate

OUT_DIR  = "matlab_equivalence_outputs"
LOG_PATH = os.path.join(OUT_DIR, "matlab_equivalence_log.txt")
CSV_PATH = os.path.join(OUT_DIR, "matlab_equivalence_results.csv")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def _base() -> Config:
    """MATLAB RF Satellite Link nominal parameters."""
    return Config(
        modulation_order=16,
        carrier_freq_hz=4.0e9,
        sat_altitude_km=35_600.0,
        tx_antenna_diameter_m=0.4,
        rx_antenna_diameter_m=0.4,
        lna_gain_db=30.0,
        noise_temp_k=20.0,           # MATLAB default noise temp
        hpa_input_backoff_db=30.0,   # MATLAB default IBO (negligible NL)
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


def run_one(cfg: Config,
            name: str,
            override_doppler_hz=None,
            override_noise_temp_k=None,
            custom_backoff_db=None,
            section: str = "A") -> ScenarioResult:
    """Execute one full Tx→Channel→Rx simulation and return metrics."""
    attach_srrc_h(cfg)
    rng = np.random.default_rng(cfg.random_seed)

    tx = transmit(cfg, rng, custom_backoff_db=custom_backoff_db)

    ch = propagate(tx.after_hpa, cfg, rng,
                   override_doppler_hz=override_doppler_hz,
                   override_noise_temp_k=override_noise_temp_k)

    # Pass true Doppler only in ideal mode (upper-bound test)
    dop_rx = (override_doppler_hz
              if (cfg.apply_doppler_correction
                  and cfg.cfo_correction_mode == "ideal")
              else None)

    rx = receive(ch.signal, tx.bits, cfg,
                 override_doppler_hz=dop_rx)

    r = ScenarioResult(name)
    r.ber      = rx.ber
    r.ser      = rx.ser
    r.n_errors = rx.n_bit_errors
    r.snr_db   = ch.snr_db
    nt = override_noise_temp_k if override_noise_temp_k is not None else cfg.noise_temp_k
    r.ebn0_db  = compute_ebn0_db(cfg, noise_temp_k=nt)
    tx_ref     = bits_to_symbols(tx.bits, cfg.modulation_order)
    r.evm_pct  = compute_evm(tx_ref, rx.symbols)
    r.papr_db  = compute_papr_db(tx.after_hpa)
    r.notes    = f"section={section}"
    return r


# ============================================================
# SECTION A — MATLAB-equivalent scenarios
# ============================================================
# Each group corresponds to one of the MATLAB Model Parameters settings.

print("=" * 70)
print("Section A: MATLAB-equivalent scenarios")
print("=" * 70)

results_A = []

# ── A1. MATLAB nominal (default parameters) ──────────────────────────────
# MATLAB defaults: 16-QAM, 35600 km, 4 GHz, 0.4m dishes, T=20K, IBO=30dB,
#                 no Doppler, no phase noise, no IQ/DC impairments.
print("\nA1. MATLAB nominal (T=20K, IBO=30dB, no impairments)")
r = run_one(_base(), "A1-Nominal (MATLAB defaults, T=20K)")
print(f"     BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%  (expect BER=0, EVM~1%)")
results_A.append(r)

# ── A2-A5. MATLAB noise-temperature sweep ────────────────────────────────
# MATLAB parameter: Noise temperature = 0, 20, 290, 500 K
print("\nA2-A5. Noise temperature sweep (MATLAB parameter choices)")
for T, label in [(0, "A2-Noise T=0K (no noise)"),
                 (20, "A3-Noise T=20K (MATLAB nominal)"),
                 (290, "A4-Noise T=290K (typical)"),
                 (500, "A5-Noise T=500K (high noise)")]:
    r = run_one(_base(), label, override_noise_temp_k=float(T))
    print(f"     {label}: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%")
    results_A.append(r)

# ── A6-A8. MATLAB HPA backoff sweep ──────────────────────────────────────
# MATLAB parameter: HPA backoff = 30 dB, 7 dB, 1 dB
print("\nA6-A8. HPA back-off sweep (MATLAB parameter choices)")
for ibo, label in [(30, "A6-HPA IBO=30dB (negligible NL)"),
                   (7,  "A7-HPA IBO=7dB (moderate NL)"),
                   (1,  "A8-HPA IBO=1dB (severe NL)")]:
    r = run_one(_base(), label, custom_backoff_db=ibo)
    print(f"     {label}: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%")
    results_A.append(r)

# ── A9. HPA bypass — MATLAB-equivalent linear amplifier baseline ──────────
print("\nA9. HPA bypass (ideal linear amplifier baseline)")
cfg = _base(); cfg.apply_hpa = False
r = run_one(cfg, "A9-HPA bypass (ideal linear)")
print(f"     BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%  (expect BER=0)")
results_A.append(r)

# ── A10-A11. MATLAB DPD scenarios ────────────────────────────────────────
# MATLAB parameter: Digital predistortion enabled/disabled with IBO=7dB
# MATLAB states DPD corrects moderate (7dB) but not severe (1dB) distortion.
print("\nA10-A11. DPD scenarios (MATLAB Digital Predistortion parameter)")
for ibo, dpd, label in [
    (7,  True,  "A10-HPA IBO=7dB WITH DPD"),
    (1,  True,  "A11-HPA IBO=1dB WITH DPD"),
]:
    cfg = _base(); cfg.apply_dpd = True
    r = run_one(cfg, label, custom_backoff_db=ibo)
    ref = ("DPD effective" if ibo == 7 else "DPD insufficient")
    print(f"     {label}: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%  ({ref})")
    results_A.append(r)

# ── A12-A13. MATLAB Doppler scenarios ────────────────────────────────────
# MATLAB parameter: Doppler = 0 Hz, 3 Hz
# Note: 3 Hz in MATLAB is 3 Hz relative to its physical symbol rate.
# In Python normalised mode, the Doppler timebase is mapped through
# Config.doppler_reference_symbol_rate_baud so the MATLAB 3 Hz setting is
# physically meaningful rather than 3 cycles/symbol.
# The corrected case uses coarse 4th-power + DD PLL approximation.
print("\nA12-A13. Doppler scenarios (MATLAB Doppler Error parameter)")
print("  NOTE: 'Doppler correction' in MATLAB uses comm.CarrierSynchronizer.")
print("  Python uses coarse-4th-power + DA-phase + fine DD-PLL (closest approximation).")
print("  In normalised amplitude mode, Doppler=3Hz is interpreted relative to")
print("  Config.doppler_reference_symbol_rate_baud so it remains a physical 3 Hz")
print("  offset, not 3 cycles/symbol.")

cfg_no = _base()
cfg_no.apply_doppler_correction = False
r = run_one(cfg_no, "A12-Doppler=3Hz NO correction", override_doppler_hz=3.0)
print(f"     A12: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%  (expect large BER / rotating constellation)")
results_A.append(r)

cfg_cs = _base()
cfg_cs.apply_doppler_correction = True
cfg_cs.cfo_correction_mode = "carrier_sync"
cfg_cs.carrier_sync_loop_bw = 0.01
cfg_cs.carrier_sync_damping = 0.707
r = run_one(cfg_cs, "A13-Doppler=3Hz carrier_sync (approx. MATLAB)", override_doppler_hz=3.0)
print(f"     A13: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%  (MATLAB-closest correction; expect BER→0)")
results_A.append(r)

# ── A14-A16. MATLAB phase noise scenarios ────────────────────────────────
# MATLAB parameter: Phase Noise = Negligible (-100 dBc/Hz), Low (-55 dBc/Hz),
#                                  High (-48 dBc/Hz)
# Python approximation: Wiener 1/f² model (not MATLAB's interpolated FIR table).
# physical_sample_rate_hz = 8 MHz (MATLAB RF Satellite Link typical rate).
print("\nA14-A16. Phase noise scenarios (MATLAB Phase Noise parameter)")
print("  APPROXIMATION: Python uses Wiener 1/f² model.")
print("  MATLAB uses an interpolated FIR PSD filter (internal implementation).")
print("  Levels match; PSD shape differs off the reference offset.")
for dbc, label in [
    (-100.0, "A14-Phase noise Negligible (-100 dBc/Hz @ 100 Hz)"),
    (-55.0,  "A15-Phase noise Low (-55 dBc/Hz @ 100 Hz)"),
    (-48.0,  "A16-Phase noise High (-48 dBc/Hz @ 100 Hz)"),
]:
    cfg = _base()
    cfg.apply_phase_noise = True
    cfg.phase_noise_use_white = False
    cfg.phase_noise_dbc_hz = dbc
    cfg.phase_noise_freq_offset_hz = 100.0
    cfg.phase_noise_physical_sample_rate_hz = 8_000_000.0
    r = run_one(cfg, label, override_noise_temp_k=0.0)
    print(f"     {label[-30:]}: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%")
    results_A.append(r)

# ── A17-A22. MATLAB I/Q imbalance scenarios ──────────────────────────────
# MATLAB parameter: I/Q imbalance = None, Amplitude (3dB), Phase (20deg),
#                   combined; with and without IQ correction.
# IQ imbalance model: EXACT match to comm.IQImbalance symmetric ± model.
# IQ compensator: closest approximation to comm.IQImbalanceCompensator (DD LMS).
print("\nA17-A22. I/Q imbalance scenarios (MATLAB I/Q Imbalance and Correction parameters)")
print("  Imbalance model: EXACT (symmetric ± comm.IQImbalance).")
print("  Compensator: APPROXIMATION (DD two-stage LMS ≈ comm.IQImbalanceCompensator).")

for amp, phase, corr, label in [
    (3.0,  0.0,  False, "A17-IQ amplitude 3dB NO correction"),
    (3.0,  0.0,  True,  "A18-IQ amplitude 3dB WITH correction (approx)"),
    (0.0,  20.0, False, "A19-IQ phase 20deg NO correction"),
    (0.0,  20.0, True,  "A20-IQ phase 20deg WITH correction (approx)"),
    (3.0,  20.0, False, "A21-IQ combined 3dB+20deg NO correction"),
    (3.0,  20.0, True,  "A22-IQ combined 3dB+20deg WITH correction (approx)"),
]:
    cfg = _base()
    cfg.apply_iq_imbalance = True
    cfg.iq_amplitude_imbalance_db = amp
    cfg.iq_phase_imbalance_deg = phase
    cfg.apply_iq_correction = corr
    r = run_one(cfg, label)
    print(f"     {label[-30:]}: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%")
    results_A.append(r)

# ── A23-A26. MATLAB DC offset scenarios ──────────────────────────────────
# MATLAB parameter: I/Q Imbalance = In-phase DC (1e-8), Quadrature DC (5e-8)
#                   DC offset correction = Enabled/Disabled
#
# DC OFFSET EQUIVALENCE EXPLANATION
# ------------------------------------
# MATLAB's raw absolute values (1e-8/5e-8 V) cannot be used directly in Python
# normalised mode because the normalised signal RMS at the injection point
# (before LNA, after path-loss) is only ~7.6e-10.  The raw values would be
# 13× and 66× the signal — completely destroying it.
#
# Normalised-equivalent fractions (derived from MATLAB documented behaviour):
#   dc_offset_i = 0.05 (5% of RMS)  ← reproduces "changes constellation, no errors"
#   dc_offset_q = 0.29 (29% of RMS) ← reproduces "causes errors even without noise"
# See config.py for the full derivation.
print("\nA23-A26. DC offset scenarios (MATLAB DC Offset parameter)")
print("  DC OFFSET NOTE: MATLAB absolute values (1e-8/5e-8) are incompatible")
print("  with Python normalised mode (would be 13-66× signal RMS).")
print("  Using derived normalised-equivalent fractions (5% / 29% of signal RMS)")
print("  that reproduce MATLAB's stated qualitative behaviour.")

for dc_i, dc_q, corr, label in [
    (0.05, 0.0,  False, "A23-DC I=5% (MATLAB 1e-8 equiv) NO correction"),
    (0.05, 0.0,  True,  "A24-DC I=5% (MATLAB 1e-8 equiv) WITH correction"),
    (0.0,  0.29, False, "A25-DC Q=29% (MATLAB 5e-8 equiv) NO correction"),
    (0.0,  0.29, True,  "A26-DC Q=29% (MATLAB 5e-8 equiv) WITH correction"),
]:
    cfg = _base()
    cfg.apply_dc_offset = True
    cfg.dc_offset_mode = "relative"
    cfg.dc_offset_i = dc_i
    cfg.dc_offset_q = dc_q
    cfg.apply_dc_correction = corr
    r = run_one(cfg, label)
    exp = ""
    if "I=5%" in label and not corr:
        exp = " (expect BER=0, EVM≈5%)"
    elif "Q=29%" in label and not corr:
        exp = " (expect BER>0, errors alone)"
    elif corr:
        exp = " (expect BER→0)"
    print(f"     {label[-30:]}: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%{exp}")
    results_A.append(r)


# ============================================================
# SECTION B — Extra validation (non-MATLAB scenarios)
# ============================================================
# These tests go beyond the MATLAB example for robustness verification.
# They are NOT claimed to be MATLAB-equivalent.

print("\n" + "=" * 70)
print("Section B: Extra validation (non-MATLAB scenarios)")
print("=" * 70)

results_B = []

# B1. Ideal Doppler correction (upper bound)
print("\nB1. Doppler ideal correction (upper bound, NOT a MATLAB block)")
cfg = _base()
cfg.apply_doppler_correction = True
cfg.cfo_correction_mode = "ideal"
r = run_one(cfg, "B1-Doppler ideal correction (extra)", override_doppler_hz=3.0,
            section="B")
print(f"     BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%  (upper bound only)")
results_B.append(r)

# B2. IQ ideal correction (mathematical inverse, NOT MATLAB compensator)
print("\nB2. IQ ideal correction (mathematical inverse, extra)")
for amp, phase, label in [
    (3.0,  0.0,  "B2a-IQ amp 3dB ideal (extra)"),
    (0.0,  20.0, "B2b-IQ phase 20deg ideal (extra)"),
    (3.0,  20.0, "B2c-IQ combined ideal (extra)"),
]:
    cfg = _base()
    cfg.apply_iq_imbalance = True
    cfg.iq_amplitude_imbalance_db = amp
    cfg.iq_phase_imbalance_deg = phase
    cfg.apply_iq_correction = False  # handled below
    attach_srrc_h(cfg)
    rng = np.random.default_rng(cfg.random_seed)
    tx = transmit(cfg, rng)
    ch = propagate(tx.after_hpa, cfg, rng)
    delay = filter_delay(cfg.span, cfg.samples_per_symbol)
    syms_mf = rx_filter(ch.signal, cfg.srrc_h, cfg.samples_per_symbol, delay)
    syms_corr = iq_correct_ideal(syms_mf, amp, phase)
    from receiver import _agc, _dc_blocker
    syms_final = _agc(syms_corr)
    rx_bits = symbols_to_bits(syms_final[:cfg.num_symbols], cfg.modulation_order)
    ber, _, _ = symbol_error_rate(tx.bits, rx_bits, cfg.bits_per_symbol)
    ref = bits_to_symbols(tx.bits, cfg.modulation_order)
    evm = compute_evm(ref, syms_final)
    r = ScenarioResult(label)
    r.ber = ber; r.evm_pct = evm; r.notes = "section=B extra"
    print(f"     {label}: BER={r.ber:.3e}  EVM={r.evm_pct:.1f}%  (validates inverse formula)")
    results_B.append(r)

# B3. Eb/N0 BER curve validation
print("\nB3. Eb/N0 BER validation (extra — not in MATLAB example)")
from filters import srrc_coeffs, tx_filter
from metrics import ber_theory_16qam_awgn

cfg_b = _base(); cfg_b.apply_hpa = False
h   = srrc_coeffs(cfg_b.rolloff, cfg_b.span, cfg_b.samples_per_symbol)
dly = filter_delay(cfg_b.span, cfg_b.samples_per_symbol)
bps = cfg_b.bits_per_symbol

print(f"     {'Eb/N0':>8}  {'BER_sim':>10}  {'BER_theory':>12}  {'ratio':>6}")
for ebn0_db in [5, 7, 9, 11, 13, 15]:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=cfg_b.num_symbols * bps)
    syms = bits_to_symbols(bits, cfg_b.modulation_order)
    tx_s = tx_filter(syms, h, cfg_b.samples_per_symbol)
    tx_s /= np.sqrt(np.mean(np.abs(tx_s)**2))
    syms_mf = rx_filter(tx_s, h, cfg_b.samples_per_symbol, dly)
    syms_mf /= np.sqrt(np.mean(np.abs(syms_mf)**2))
    sigma = np.sqrt(1.0 / (10**(ebn0_db/10) * bps))
    noise = sigma * (rng.standard_normal(len(syms_mf)) + 1j*rng.standard_normal(len(syms_mf)))
    rx_b = symbols_to_bits((syms_mf + noise)[:cfg_b.num_symbols], cfg_b.modulation_order)
    ber_s, _, _ = symbol_error_rate(bits, rx_b, bps)
    ber_t = float(ber_theory_16qam_awgn(np.array([ebn0_db]))[0])
    ratio = ber_s/ber_t if ber_t > 0 and ber_s > 0 else float('nan')
    r_label = f"{ratio:.2f}" if not (isinstance(ratio, float) and np.isnan(ratio)) else "  —"
    print(f"     {ebn0_db:>8} dB  {ber_s:>10.3e}  {ber_t:>12.3e}  {r_label:>6}")
    sr = ScenarioResult(f"B3-Eb/N0={ebn0_db}dB"); sr.ber = ber_s; sr.notes = "section=B extra"
    results_B.append(sr)


# ============================================================
# Print and save
# ============================================================

all_results = results_A + results_B

HDR = f"{'Scenario':<56} {'BER':>9} {'Eb/N0':>8} {'EVM%':>7}"
SEP = "-" * (len(HDR))
lines = [
    "=" * len(HDR),
    "RF Satellite Link – MATLAB Equivalence Results",
    "Reference: https://www.mathworks.com/help/comm/ug/rf-satellite-link.html",
    "=" * len(HDR),
    "SECTION A: MATLAB-equivalent scenarios",
    SEP,
    HDR,
    SEP,
]
for r in results_A:
    lines.append(f"  {r.name:<54} {r.ber:>9.3e} {r.ebn0_db:>8.1f} {r.evm_pct:>7.1f}")

lines += [SEP, "SECTION B: Extra validation (non-MATLAB)", SEP, HDR, SEP]
for r in results_B:
    ber_s = f"{r.ber:.3e}" if isinstance(r.ber, float) else str(r.ber)
    lines.append(f"  {r.name:<54} {r.ber:>9.3e} {'':>8} {r.evm_pct:>7.1f}")
lines.append("=" * len(HDR))

report = "\n".join(lines)
print("\n" + report)

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write(report + "\n")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Section", "Scenario", "BER", "SER", "EbN0_dB",
                "SNR_dB", "EVM_pct", "PAPR_dB", "Notes"])
    for r in all_results:
        section = r.notes.split("=")[1].split()[0] if "section=" in r.notes else "A"
        # Section B rows use shortcut paths that do not compute SER properly
        # (the ScenarioResult default of 1.0 would be misleading).
        # Emit "nan" for those rows so readers know SER was not measured.
        ser_str = f"{r.ser:.6e}" if section == "A" else "nan"
        w.writerow([section, r.name, f"{r.ber:.6e}", ser_str,
                    f"{r.ebn0_db:.2f}", f"{r.snr_db:.2f}",
                    f"{r.evm_pct:.2f}", f"{r.papr_db:.2f}", r.notes])

print(f"\nResults saved to:\n  {LOG_PATH}\n  {CSV_PATH}")
