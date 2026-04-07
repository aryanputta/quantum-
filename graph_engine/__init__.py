"""
graph_engine — network loading, preprocessing, and traffic management.

Public API
----------
NetworkLoader
    Build or load a NetworkX DiGraph from various data sources.
GraphPreprocessor
    Normalise edge attributes, extract graph features, and derive
    routing-pair workloads.
TrafficManager
    Generate traffic demand matrices and update graph congestion state.
"""

from graph_engine.loader import NetworkLoader
from graph_engine.preprocessor import GraphPreprocessor
from graph_engine.traffic import TrafficManager

__all__ = [
    "NetworkLoader",
    "GraphPreprocessor",
    "TrafficManager",
]
