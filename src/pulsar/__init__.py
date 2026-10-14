"""PULSAR: superposition-decode inference engine — verification package.

Modules mirror the architecture:
  M1/M2  field.py      candidate generator + superposition field
  M3     verifier.py   exact rejection-sampling verifier
  M4a    rotor.py      rotating readout (stratified sweep) + circulant mixing
  M4c    confidence.py leaky-integrator / logistic confidence field
  M4b    pulse.py      pulse generator (water-level block selection, theta rule)
  M4d    collapse.py   Born/temperature collapse operator
  M5     engine.py     commit/emit + end-to-end orchestrator
  M6     engine.PulseClock  pulse clock (deterministic tick budget)
  M7     calibrator.py learned threshold (fixed point + REINFORCE bandit)
"""
__version__ = "0.1.0"
