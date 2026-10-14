"""
Models package for Memento-Agri.
Provides baseline machine learning models for climate risk classification
and crop yield prediction.
"""
from models.risk_model import train_risk_model
from models.yield_model import train_yield_model

__all__ = ["train_risk_model", "train_yield_model"]
