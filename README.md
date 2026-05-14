# RF Satellite Link Simulation – Python
### A faithful Python reproduction of the MATLAB RF Satellite Link example

```
satellite_link/
├── config.py        Physical constants, all tunable parameters, link-budget
├── modulation.py    Gray-coded 16-QAM mapper / hard-decision demapper
├── filters.py       Square-root raised-cosine (SRRC) filter design & application
├── impairments.py   All RF impairments: Saleh TWTA HPA, path loss, Doppler,
│                    AWGN, phase noise, I/Q imbalance, DC offset, LNA gain
├── transmitter.py   Full Tx chain: bits → symbols → SRRC → HPA
├── channel.py       Downlink: path loss + Doppler + noise + optional impairments
├── receiver.py      Full Rx chain: CFO correction → SRRC → DC/AGC/IQ → demod
├── metrics.py       BER theory (16-QAM AWGN), EVM, PAPR, link budget
├── plots.py         Constellation, spectrum, BER curve, HPA AM/AM plots
├── main.py          Experiment runner – 13 scenarios + BER vs Eb/N0 sweep
└── requirements.txt numpy, scipy, matplotlib
```

## Quick start
```bash
pip install -r requirements.txt
python main.py
# Plots saved to output_figures/
```

## Simulation chain

```
[Random bits]
     │
     ▼ bits_to_symbols()
[16-QAM symbols]  ──────────────────────► constellation "Before HPA"
     │
     ▼ tx_filter()  [SRRC upsample × 8]
[Oversampled waveform]
     │
     ▼ saleh_hpa()  [Saleh TWTA model]
[After-HPA waveform]  ──────────────────► constellation "After HPA"
     │
     ▼ apply_path_loss()  [195.5 dB FSPL – 2×21.9 dBi antenna gain]
[Attenuated signal]
     │
     ▼ apply_doppler()  [complex rotation]
     ▼ apply_lna_gain()  [30 dB]
     ▼ add_awgn_noise()  [k_B × T × B]
     ▼ add_phase_noise()  [optional]
     ▼ apply_iq_imbalance()  [optional]
     ▼ add_dc_offset()  [optional, signal-relative]
     │
     ▼ _doppler_correction()  [4th-power NDA CFO estimator]
     ▼ rx_filter()  [SRRC matched filter + downsample]
     ▼ _dc_correction()  [subtract mean]
     ▼ _agc()  [scale to unit power]
     ▼ _iq_imbalance_correction()  [2nd-order statistics]
     │
     ▼ symbols_to_bits()  [nearest-neighbour decision]
[Decoded bits]  ──────────────────────────► BER / SER
```

## Scenarios & expected results

| # | Scenario | BER |
|---|----------|-----|
| 1 | Clean (no noise, IBO=30 dB) | 0 |
| 2 | T=290 K (SNR=52 dB) | 0 |
| 3 | T=20 K  (SNR=64 dB) | 0 |
| 4 | T=500 K (SNR=50 dB) | 0 |
| 5 | Doppler 3 Hz, **no correction** | ~50% |
| 6 | Doppler 3 Hz, **corrected** | 0 |
| 7 | HPA IBO=30 dB (near-linear) | 0 |
| 8 | HPA IBO=7 dB (moderate) | ~7% |
| 9 | HPA IBO=1 dB (heavy clipping) | ~18% |
|10 | I/Q imbalance 1 dB / 5°, no corr. | 0* |
|11 | I/Q imbalance, corrected | 0 |
|12 | Phase noise σ²=1e-3 rad² | 0 |
|13 | DC offset 2% / 1.5%, corrected | 0 |

*I/Q imbalance at 52 dB SNR is below the distortion floor for 16-QAM.

## Key design choices

### SRRC filter delay
Two `np.convolve(..., mode='full')` calls (one Tx, one Rx) each prepend
`span×sps` leading samples, so the correct downsampling offset is
`2 × span × sps` (= 160 samples for the default config).

### Noise scaling
Physical noise power `k_B × T × B` is computed in SI watts, and the
simulation signal has been scaled by the physical link gains via
`apply_path_loss` and `apply_lna_gain`.  Both quantities live in the
same "simulation amplitude = physical voltage" domain, so their ratio
gives the correct SNR without any additional normalisation factor.

### DC offset
Applied as a **fraction of the signal RMS** so it is meaningful across
the huge dynamic range of the link budget (signal power ≈ 10⁻¹⁵ W after
the LNA).  Default values 0.02 (I) and 0.015 (Q) represent 2 % / 1.5 %
of RMS amplitude.

### Doppler CFO estimator
The 4th-power NDA estimator raises the signal to the 4th power to remove
the 16-QAM modulation (all 16-QAM symbols lie on a circle of radius |s|,
so s⁴ collapses to a tone at 4×f_doppler).  The spectral peak frequency
is then divided by 4 to recover the CFO estimate.
