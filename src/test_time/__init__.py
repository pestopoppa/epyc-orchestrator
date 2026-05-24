"""Test-time-compute techniques (P21).

Per-request methods that trade extra inference for accuracy, applied above
role-routing. Currently houses the DeepConf offline scorer (P21.A, intake-603).
The OptiLLM-style method-selection axis (P21.B) is the planned future neighbour.

See research/deep-dives/optillm-test-time-techniques.md for the analysis and the
autopilot-scope determination (build here in a dedicated session; autopilot only
sweeps the knob surface once it exists).
"""
