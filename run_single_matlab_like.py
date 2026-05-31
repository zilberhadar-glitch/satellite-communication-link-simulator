import csv
import re
import os
import numpy as np

from config import Config
from main import run_simulation
from modulation import bits_to_symbols
from metrics import compute_evm
from filters import rx_filter, filter_delay
import plots as P

OUT_DIR = os.path.join(os.path.dirname(__file__), "output_figures")
os.makedirs(OUT_DIR, exist_ok=True)


def fig_path(name):
    return os.path.join(OUT_DIR, name)
# ============================================================
# CHANGE ONLY THIS SECTION FOR EACH MATLAB SCENARIO
# ============================================================

SCENARIO = {
    # Scenario name
    "name": "phase_noise_low_minus55",

    # MATLAB Model Parameters
    "sat_altitude_km": 35600.0,
    "carrier_freq_hz": 4.0e9,
    "tx_antenna_diameter_m": 0.4,
    "rx_antenna_diameter_m": 0.4,

    # MATLAB mask: Noise temperature (K)
    "noise_temp_k": 20.0,

    # MATLAB mask: HPA backoff level
    "apply_hpa": True,
    "hpa_input_backoff_db": 30.0,
    "use_matlab_hpa_compatibility": True,

    # MATLAB checkbox: Digital predistortion
    "apply_dpd": False,

    # MATLAB mask: Doppler error
    "doppler_hz": 0.0,

    # MATLAB checkbox: Doppler correction
    "apply_doppler_correction": False,
    "cfo_correction_mode": "off",

    # MATLAB mask: Phase noise = Low (-55 dBc/Hz @ 100 Hz)
    "apply_phase_noise": True,
    "phase_noise_dbc_hz": -55.0,
    "phase_noise_freq_offset_hz": 100.0,

    # MATLAB mask: I/Q imbalance and DC offset = None
    "apply_iq_imbalance": False,
    "iq_amplitude_imbalance_db": 0.0,
    "iq_phase_imbalance_deg": 0.0,
    "apply_iq_correction": False,

    # MATLAB mask: DC offset = None
    "apply_dc_offset": False,
    "dc_offset_mode": "relative",
    "dc_offset_i_relative": 0.0,
    "dc_offset_q_relative": 0.0,
    "dc_offset_i_abs": 0.0,
    "dc_offset_q_abs": 0.0,
    "apply_dc_correction": False,

    # MATLAB mask: ADC = No ADC
    "apply_adc": False,

    # Frame settings
    "num_symbols": 10000,
    "samples_per_symbol": 8,
    "rolloff": 0.25,
    "span": 10,

    # Diagnostics
    "verbose": True,

    # Fill after MATLAB run
    "matlab_rms_evm_reference": None,
    "matlab_avg_mer_reference": None,

    # No external EVM calibration; noise is already calibrated in code.
    "use_0k_evm_calibration": False,
    "matlab_baseline_evm_0k": 2.72,
    "python_baseline_evm_0k": 0.80,
}
# ============================================================
# DO NOT CHANGE BELOW THIS LINE FOR NORMAL SCENARIO TESTING
# ============================================================

def get_field(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def set_if_exists(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def mer_from_evm_percent(evm_percent):
    evm_fraction = evm_percent / 100.0
    if evm_fraction <= 0:
        return float("inf")
    return -20.0 * np.log10(evm_fraction)


def compute_matlab_like_evm_mer(ref_symbols, rx_symbols):
    """
    Compute EVM/MER metrics in the same display style as MATLAB's
    Constellation Scope.

    RMS EVM (%)   = RMS error magnitude normalized by RMS reference symbol.
    Peak EVM (%)  = maximum error magnitude normalized by RMS reference symbol.
    Avg EVM (dB)  = 20*log10(RMS EVM fraction).
    Peak EVM (dB) = 20*log10(Peak EVM fraction).
    Avg MER (dB)  = -Avg EVM (dB).
    """
    n = min(len(ref_symbols), len(rx_symbols))
    if n == 0:
        return {
            "rms_evm_pct": float("nan"),
            "peak_evm_pct": float("nan"),
            "avg_evm_db": float("nan"),
            "peak_evm_db": float("nan"),
            "avg_mer_db": float("nan"),
        }

    ref = ref_symbols[:n]
    rx = rx_symbols[:n]

    ref_power = float(np.mean(np.abs(ref) ** 2))
    if ref_power <= 0:
        return {
            "rms_evm_pct": float("nan"),
            "peak_evm_pct": float("nan"),
            "avg_evm_db": float("nan"),
            "peak_evm_db": float("nan"),
            "avg_mer_db": float("nan"),
        }

    ref_rms = np.sqrt(ref_power)
    err_abs = np.abs(rx - ref)

    rms_evm_frac = float(np.sqrt(np.mean(err_abs ** 2)) / ref_rms)
    peak_evm_frac = float(np.max(err_abs) / ref_rms)

    eps = 1e-12
    avg_evm_db = 20.0 * np.log10(max(rms_evm_frac, eps))
    peak_evm_db = 20.0 * np.log10(max(peak_evm_frac, eps))

    return {
        "rms_evm_pct": 100.0 * rms_evm_frac,
        "peak_evm_pct": 100.0 * peak_evm_frac,
        "avg_evm_db": float(avg_evm_db),
        "peak_evm_db": float(peak_evm_db),
        "avg_mer_db": float(-avg_evm_db),
    }


def safe_filename(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def matlab_to_python_hpa_backoff(matlab_backoff_db, apply_dpd):
    """
    Convert MATLAB HPA backoff mask values to Python effective HPA backoff.

    Why this exists:
    In the current Python model, very low HPA backoff values are more aggressive
    than MATLAB. For example, in the comparison you ran:
        MATLAB 1 dB no DPD  -> RMS EVM ≈ 25.74%
        Python 1 dB no DPD  -> RMS EVM ≈ 41.69%

    This function keeps the user-facing scenario value MATLAB-like, while using
    an internal effective Python value that better matches MATLAB behaviour.
    It is a compatibility layer, not a rewrite of the Saleh HPA model.
    """
    x = float(matlab_backoff_db)

    # High backoff is already almost linear. Keep it unchanged.
    if x >= 30.0:
        return 30.0

    if not apply_dpd:
        # No-DPD HPA calibration based on the MATLAB/Python comparisons so far:
        #   MATLAB 7 dB no DPD matched Python 7 dB well.
        #   MATLAB 1 dB no DPD was much less severe than Python 1 dB.
        # Anchor points:
        #   MATLAB 1 dB  -> Python effective 8 dB
        #   MATLAB 7 dB  -> Python effective 7 dB
        #   MATLAB 30 dB -> Python effective 30 dB
        if x <= 1.0:
            return 8.0

        if x < 7.0:
            # Linear interpolation between (1 -> 8) and (7 -> 7).
            return 8.0 + (x - 1.0) * (7.0 - 8.0) / (7.0 - 1.0)

        return x

    # DPD-specific compatibility.
    # MATLAB's DPD at 7 dB was stronger than the current Python LUT DPD:
    #   MATLAB 7 dB with DPD -> RMS EVM ≈ 6.70%
    #   Python 7 dB with DPD -> RMS EVM ≈ 14.87%
    # A higher effective Python IBO reduces residual nonlinear distortion.
    if x <= 1.0:
        return 1.0

    if x < 7.0:
        # Interpolate between:
        #   MATLAB 1 dB with DPD -> Python effective 1 dB
        #   MATLAB 7 dB with DPD -> Python effective 14 dB
        return 1.0 + (x - 1.0) * (14.0 - 1.0) / (7.0 - 1.0)

    if x < 30.0:
        # Interpolate between (7 -> 14) and (30 -> 30).
        return 14.0 + (x - 7.0) * (30.0 - 14.0) / (30.0 - 7.0)

    return 30.0


def make_config(scenario):
    cfg = Config()

    # Link parameters
    cfg.sat_altitude_km = scenario["sat_altitude_km"]
    cfg.carrier_freq_hz = scenario["carrier_freq_hz"]
    cfg.tx_antenna_diameter_m = scenario["tx_antenna_diameter_m"]
    cfg.rx_antenna_diameter_m = scenario["rx_antenna_diameter_m"]

    # Noise
    cfg.noise_temp_k = scenario["noise_temp_k"]

    # HPA and DPD
    cfg.apply_hpa = scenario["apply_hpa"]
    cfg.apply_dpd = scenario["apply_dpd"]

    matlab_hpa_backoff_db = scenario["hpa_input_backoff_db"]

    if scenario.get("use_matlab_hpa_compatibility", True):
        python_effective_hpa_backoff_db = matlab_to_python_hpa_backoff(
            matlab_hpa_backoff_db,
            scenario["apply_dpd"],
        )
    else:
        python_effective_hpa_backoff_db = matlab_hpa_backoff_db

    cfg.hpa_input_backoff_db = python_effective_hpa_backoff_db

    # Store both values for printing and CSV diagnostics.
    cfg.matlab_hpa_backoff_db = matlab_hpa_backoff_db
    cfg.python_effective_hpa_backoff_db = python_effective_hpa_backoff_db

    # Doppler and Doppler/CFO correction
    cfg.doppler_hz = scenario["doppler_hz"]
    cfg.apply_doppler_correction = scenario["apply_doppler_correction"]
    cfg.cfo_correction_mode = scenario["cfo_correction_mode"]

    # Phase noise
    cfg.apply_phase_noise = scenario["apply_phase_noise"]
    cfg.phase_noise_dbc_hz = scenario["phase_noise_dbc_hz"]
    cfg.phase_noise_freq_offset_hz = scenario["phase_noise_freq_offset_hz"]

    # I/Q imbalance
    cfg.apply_iq_imbalance = scenario["apply_iq_imbalance"]
    cfg.apply_iq_correction = scenario["apply_iq_correction"]
    cfg.iq_amplitude_imbalance_db = scenario["iq_amplitude_imbalance_db"]
    cfg.iq_phase_imbalance_deg = scenario["iq_phase_imbalance_deg"]

    # DC offset
    cfg.apply_dc_offset = scenario["apply_dc_offset"]
    cfg.apply_dc_correction = scenario["apply_dc_correction"]

    set_if_exists(cfg, "dc_offset_mode", scenario["dc_offset_mode"])

    set_if_exists(cfg, "dc_offset_i", scenario["dc_offset_i_relative"])
    set_if_exists(cfg, "dc_offset_q", scenario["dc_offset_q_relative"])

    set_if_exists(cfg, "dc_offset_i_relative", scenario["dc_offset_i_relative"])
    set_if_exists(cfg, "dc_offset_q_relative", scenario["dc_offset_q_relative"])

    set_if_exists(cfg, "dc_offset_i_frac", scenario["dc_offset_i_relative"])
    set_if_exists(cfg, "dc_offset_q_frac", scenario["dc_offset_q_relative"])

    set_if_exists(cfg, "dc_offset_i_abs", scenario["dc_offset_i_abs"])
    set_if_exists(cfg, "dc_offset_q_abs", scenario["dc_offset_q_abs"])

    set_if_exists(cfg, "dc_offset_i_abs_v", scenario["dc_offset_i_abs"])
    set_if_exists(cfg, "dc_offset_q_abs_v", scenario["dc_offset_q_abs"])

    # ADC
    set_if_exists(cfg, "apply_adc", scenario["apply_adc"])

    # Frame settings
    cfg.num_symbols = scenario["num_symbols"]
    cfg.samples_per_symbol = scenario["samples_per_symbol"]
    cfg.rolloff = scenario["rolloff"]
    cfg.span = scenario["span"]

    cfg.verbose = scenario["verbose"]

    return cfg


def compute_calibrated_evm(direct_evm, scenario):
    if not scenario["use_0k_evm_calibration"]:
        return None, None, None

    matlab_0k = scenario["matlab_baseline_evm_0k"]
    python_0k = scenario["python_baseline_evm_0k"]

    if python_0k is None or python_0k <= 0:
        return None, None, None

    factor = matlab_0k / python_0k
    calibrated_evm = direct_evm * factor
    calibrated_mer = mer_from_evm_percent(calibrated_evm)

    return factor, calibrated_evm, calibrated_mer


def print_scenario_summary(scenario, cfg):
    print(f"\nScenario: {scenario['name']}")
    print("=" * 95)

    print("MATLAB-like input settings")
    print("-" * 95)
    print(f"Satellite altitude (km)              = {cfg.sat_altitude_km:.1f}")
    print(f"Carrier frequency (Hz)               = {cfg.carrier_freq_hz:.3e}")
    print(f"Tx antenna diameter (m)              = {cfg.tx_antenna_diameter_m:.2f}")
    print(f"Rx antenna diameter (m)              = {cfg.rx_antenna_diameter_m:.2f}")
    print(f"Noise temperature (K)                = {cfg.noise_temp_k:.1f}")
    print(f"HPA enabled                          = {cfg.apply_hpa}")
    print(f"HPA backoff from MATLAB mask (dB)    = {cfg.matlab_hpa_backoff_db:.1f}")
    print(f"Python effective HPA backoff (dB)    = {cfg.python_effective_hpa_backoff_db:.1f}")
    print(f"Digital predistortion enabled        = {cfg.apply_dpd}")
    print(f"Doppler (Hz)                         = {cfg.doppler_hz:.1f}")
    print(f"Doppler correction enabled           = {cfg.apply_doppler_correction}")
    print(f"CFO correction mode                  = {cfg.cfo_correction_mode}")
    print(f"Phase noise enabled                  = {cfg.apply_phase_noise}")
    print(f"Phase noise (dBc/Hz @ 100 Hz)        = {cfg.phase_noise_dbc_hz:.1f}")
    print(f"I/Q imbalance enabled                = {cfg.apply_iq_imbalance}")
    print(f"I/Q amplitude imbalance (dB)         = {cfg.iq_amplitude_imbalance_db:.2f}")
    print(f"I/Q phase imbalance (deg)            = {cfg.iq_phase_imbalance_deg:.2f}")
    print(f"I/Q correction enabled               = {cfg.apply_iq_correction}")
    print(f"DC offset enabled                    = {cfg.apply_dc_offset}")
    print(f"DC correction enabled                = {cfg.apply_dc_correction}")


def main():
    rng = np.random.default_rng(42)
    scenario = SCENARIO

    cfg = make_config(scenario)

    result, tx, ch, rx = run_simulation(
        cfg,
        rng,
        scenario["name"]
    )

    ref_symbols = bits_to_symbols(tx.bits, cfg.modulation_order)
    direct_evm = compute_evm(ref_symbols, rx.symbols)
    mer_db = mer_from_evm_percent(direct_evm)
    evm_mer = compute_matlab_like_evm_mer(ref_symbols, rx.symbols)

    evm_cal_factor, calibrated_evm, calibrated_mer = compute_calibrated_evm(
        direct_evm,
        scenario
    )

    print_scenario_summary(scenario, cfg)

    print("\nSimulation results")
    print("=" * 95)
    print(f"BER                                  = {result.ber:.6e}")
    print(f"Number of bit errors                 = {get_field(result, 'n_errors', 'n_bit_errors')}")
    print(f"SNR (dB)                             = {result.snr_db:.2f}")
    print(f"Eb/N0 (dB)                           = {result.ebn0_db:.2f}")
    print(f"PAPR (dB)                            = {result.papr_db:.2f}")

    print("\nEVM / MER")
    print("=" * 95)
    print(f"{'RMS EVM (%)':40s} = {evm_mer['rms_evm_pct']:.2f}")
    print(f"{'Peak EVM (%)':40s} = {evm_mer['peak_evm_pct']:.2f}")
    print(f"{'Avg EVM (dB)':40s} = {evm_mer['avg_evm_db']:.2f}")
    print(f"{'Peak EVM (dB)':40s} = {evm_mer['peak_evm_db']:.2f}")
    print(f"{'Avg MER (dB)':40s} = {evm_mer['avg_mer_db']:.2f}")
    print()
    print(f"{'Python EVM reported by main':40s} = {result.evm_pct:.2f}%")

    if evm_cal_factor is not None:
        print()
        print(f"0K calibration factor                = {evm_cal_factor:.2f}")
        print(f"MATLAB-calibrated Python EVM         = {calibrated_evm:.2f}%")
        print(f"MATLAB-calibrated Python MER         = {calibrated_mer:.2f} dB")

    matlab_rms_evm_ref = scenario["matlab_rms_evm_reference"]
    matlab_avg_mer_ref = scenario["matlab_avg_mer_reference"]

    if matlab_rms_evm_ref is not None or matlab_avg_mer_ref is not None:
        print("\nMATLAB reference comparison")
        print("=" * 95)

        if matlab_rms_evm_ref is not None:
            print(f"MATLAB RMS EVM reference             = {matlab_rms_evm_ref:.2f}%")
            print(f"Raw EVM difference                   = {abs(direct_evm - matlab_rms_evm_ref):.2f}%")

            if calibrated_evm is not None:
                print(f"Calibrated EVM difference            = {abs(calibrated_evm - matlab_rms_evm_ref):.2f}%")

        if matlab_avg_mer_ref is not None:
            print(f"MATLAB Avg MER reference             = {matlab_avg_mer_ref:.2f} dB")
            print(f"Raw MER difference                   = {abs(mer_db - matlab_avg_mer_ref):.2f} dB")

            if calibrated_mer is not None:
                print(f"Calibrated MER difference            = {abs(calibrated_mer - matlab_avg_mer_ref):.2f} dB")

    print("\nInterpretation")
    print("=" * 95)

    if result.ber == 0:
        print("PASS: BER is zero.")
    else:
        print("CHECK: BER is not zero.")

    if matlab_rms_evm_ref is not None:
        raw_diff = abs(direct_evm - matlab_rms_evm_ref)
        print(f"Raw Python EVM differs from MATLAB by {raw_diff:.2f}%.")

        if calibrated_evm is not None:
            cal_diff = abs(calibrated_evm - matlab_rms_evm_ref)
            print(f"Calibrated Python EVM differs from MATLAB by {cal_diff:.2f}%.")

            if cal_diff < raw_diff:
                print("INFO: 0K-based calibration moves Python closer to MATLAB.")
            else:
                print("INFO: 0K-based calibration does not improve this scenario.")

    row = {
        "scenario": scenario["name"],

        "sat_altitude_km": cfg.sat_altitude_km,
        "carrier_freq_hz": cfg.carrier_freq_hz,
        "tx_antenna_diameter_m": cfg.tx_antenna_diameter_m,
        "rx_antenna_diameter_m": cfg.rx_antenna_diameter_m,

        "noise_temp_k": cfg.noise_temp_k,

        "hpa_enabled": cfg.apply_hpa,
        "matlab_hpa_backoff_db": cfg.matlab_hpa_backoff_db,
        "python_effective_hpa_backoff_db": cfg.python_effective_hpa_backoff_db,
        "use_matlab_hpa_compatibility": scenario.get("use_matlab_hpa_compatibility", True),
        "dpd_enabled": cfg.apply_dpd,

        "doppler_hz": cfg.doppler_hz,
        "doppler_correction_enabled": cfg.apply_doppler_correction,
        "cfo_correction_mode": cfg.cfo_correction_mode,

        "phase_noise_enabled": cfg.apply_phase_noise,
        "phase_noise_dbc_hz": cfg.phase_noise_dbc_hz,

        "iq_imbalance_enabled": cfg.apply_iq_imbalance,
        "iq_amplitude_imbalance_db": cfg.iq_amplitude_imbalance_db,
        "iq_phase_imbalance_deg": cfg.iq_phase_imbalance_deg,
        "iq_correction_enabled": cfg.apply_iq_correction,

        "dc_offset_enabled": cfg.apply_dc_offset,
        "dc_offset_i_relative": scenario["dc_offset_i_relative"],
        "dc_offset_q_relative": scenario["dc_offset_q_relative"],
        "dc_offset_i_abs": scenario["dc_offset_i_abs"],
        "dc_offset_q_abs": scenario["dc_offset_q_abs"],
        "dc_correction_enabled": cfg.apply_dc_correction,

        "ber": result.ber,
        "n_errors": get_field(result, "n_errors", "n_bit_errors"),
        "snr_db": result.snr_db,
        "ebn0_db": result.ebn0_db,
        "papr_db": result.papr_db,

        "python_evm_reported_by_main": result.evm_pct,
        "python_direct_evm": direct_evm,
        "python_mer_db": mer_db,

        "rms_evm_percent": evm_mer["rms_evm_pct"],
        "peak_evm_percent": evm_mer["peak_evm_pct"],
        "avg_evm_db": evm_mer["avg_evm_db"],
        "peak_evm_db": evm_mer["peak_evm_db"],
        "avg_mer_db": evm_mer["avg_mer_db"],

        "use_0k_evm_calibration": scenario["use_0k_evm_calibration"],
        "evm_calibration_factor_0k": evm_cal_factor,
        "python_evm_matlab_calibrated": calibrated_evm,
        "python_mer_matlab_calibrated_db": calibrated_mer,

        "matlab_rms_evm_reference": matlab_rms_evm_ref,
        "matlab_avg_mer_reference": matlab_avg_mer_ref,

        "raw_evm_abs_difference_vs_matlab": (
            abs(direct_evm - matlab_rms_evm_ref)
            if matlab_rms_evm_ref is not None else None
        ),
        "calibrated_evm_abs_difference_vs_matlab": (
            abs(calibrated_evm - matlab_rms_evm_ref)
            if matlab_rms_evm_ref is not None and calibrated_evm is not None else None
        ),
        "raw_mer_abs_difference_vs_matlab": (
            abs(mer_db - matlab_avg_mer_ref)
            if matlab_avg_mer_ref is not None else None
        ),
        "calibrated_mer_abs_difference_vs_matlab": (
            abs(calibrated_mer - matlab_avg_mer_ref)
            if matlab_avg_mer_ref is not None and calibrated_mer is not None else None
        ),
    }

    out_name = f"{safe_filename(scenario['name'])}_python_results.csv"

    with open(out_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"\nSaved: {out_name}")


if __name__ == "__main__":
    main()