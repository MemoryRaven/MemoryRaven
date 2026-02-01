"""Advanced graph features including GNNs and inference."""

from .graph_neural_networks import GraphNeuralNetwork, GNNTrainer
from .relationship_inference import RelationshipInferenceEngine
from .temporal_analysis import TemporalGraphAnalyzer

__all__ = [
    "GraphNeuralNetwork",
    "GNNTrainer",
    "RelationshipInferenceEngine",
    "TemporalGraphAnalyzer",
]