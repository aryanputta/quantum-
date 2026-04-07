"""
qubo_formulation package
========================

Converts telecom routing problems into QUBO (Quadratic Unconstrained Binary
Optimization) matrices suitable for quantum and quantum-inspired solvers.

Public API
----------
QUBOBuilder
    Constructs QUBO matrices for single- and multi-commodity routing problems.
QUBOResult
    Dataclass holding the QUBO matrix, variable map, and metadata.
"""

from .qubo_builder import QUBOBuilder, QUBOResult

__all__ = ["QUBOBuilder", "QUBOResult"]
