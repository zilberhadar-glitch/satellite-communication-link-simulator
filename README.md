# RF Satellite Link Simulation — MATLAB-to-Python Conversion

This project is a Python conversion of the MathWorks **RF Satellite Link** MATLAB/Simulink example:

https://www.mathworks.com/help/comm/ug/rf-satellite-link.html

The main script is `main.py`. It implements the satellite downlink chain and produces the main plots in `output_figures/`.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

Optional validation:

```bash
python run_matlab_equivalence_tests.py
```

## Main MATLAB-equivalent signal chain

### Transmitter

1. Bernoulli bit source
2. 16-QAM modulation
3. Square-root raised-cosine transmit filter
4. Optional digital predistortion (DPD)
5. Saleh TWTA HPA
6. Transmit antenna gain

### Downlink / receiver RF path

1. Free-space path loss
2. Doppler frequency offset
3. Receiver antenna gain
4. Receiver thermal noise
5. Phase noise
6. I/Q imbalance and DC offset
7. LNA gain

### Receiver baseband processing

1. Square-root raised-cosine receive filter
2. DC blocker
3. AGC
4. I/Q imbalance compensator
5. Carrier synchronizer / Doppler correction
6. 16-QAM demodulation
7. BER, EVM and PAPR measurements

## MATLAB-equivalent scenarios in `main.py`

The script tests the main parameter choices from the MATLAB example:

- noise temperature: 0 K, 20 K, 290 K, 500 K
- HPA input back-off: 30 dB, 7 dB, 1 dB
- DPD enabled for nonlinear HPA cases
- Doppler error: 3 Hz, with and without correction
- phase noise: -100, -55, -48 dBc/Hz at 100 Hz
- I/Q imbalance: 3 dB amplitude, 20 degree phase, and combined case
- DC offset cases equivalent to MATLAB's I and Q DC offsets in normalised mode

## Important implementation notes

Some MathWorks blocks are closed/proprietary or stateful Simulink blocks. The Python implementation therefore uses documented approximations where a 1:1 internal clone is not available:

- `comm.PhaseNoise`: implemented as a 1/f² Wiener phase-noise approximation using the same MATLAB phase-noise levels.
- `comm.AGC`: implemented as static normalisation, which is equivalent for stationary channel cases used here.
- `comm.IQImbalanceCompensator`: implemented as a two-stage decision-directed compensator.
- `comm.CarrierSynchronizer`: implemented using a coarse 4th-power estimate plus a fine decision-directed PLL.
- DC offsets: MATLAB specifies physical volt offsets; in the Python normalised signal model, the main path uses normalised-equivalent fractions that reproduce the documented MATLAB behaviour.

These approximations are documented in the source code and in `matlab_equivalence_fixes_report.md`.

## Output files

Running `python main.py` creates:

```text
output_figures/
  ber_comparison.png
  ber_vs_ebn0.png
  constellations_ibo1.png
  constellations_ibo7.png
  hpa_characteristics.png
  spectra_nominal.png
  srrc_response.png
```
