# Sem-MIR minimal simulation code

A runnable 128-byte, rate-5/6, 16-QAM MIMO transmission with iterative
MMSE-PIC/BCJR decoding. LM remains a placeholder.

```text
python -m pip install -r requirements.txt
python -B simulation_runner.py
```

The script transmits the bundled sample packet through a 4x4 Rayleigh
channel at 7 dB payload Eb/N0, runs six receiver iterations, and prints the recovered
bytes. `run(payload, ebn0_db=7.0, seed=0, config=SystemConfig())` accepts a
128-byte payload and returns bytes. Configuration is supplied in Python; there
is no command-line argument interface.

The six Python files contain the runner, receiver, convolutional codec,
MMSE-PIC detector, symbol mapping/configuration, and LM placeholder.

The conventional pipeline follows the project's Python adaptation of the
[Studer SISO simulator](https://www.csl.cornell.edu/~studer/software/SISO_sim_v1.01.zip).
`data/paper128_test.txt` contains only the 128-byte sample used by the runner.
Sample attribution and redistribution terms are in
[CORPUS_SOURCE_AND_LICENSE.md](data/CORPUS_SOURCE_AND_LICENSE.md).
