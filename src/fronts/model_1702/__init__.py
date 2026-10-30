"""Self-contained harness for evaluating the AIES FrontFinder model (model_1702) in the 2.0 paradigm.

This package loads the legacy Keras 2.10 HDF5 checkpoint under Keras 3, rebuilds its exact
input pipeline (10 legacy variables x 5 levels, legacy units and normalization), and runs it
through the repository's standard evaluation and plotting machinery so it can be compared
head-to-head with current models. It intentionally lives apart from the core library and
imports from it without modifying it.
"""
