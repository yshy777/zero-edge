import torch
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    
    # Main paths (relative to project root)
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RESULT_DIR: str = os.path.join(BASE_DIR, "experiment")
    
    # Paths to specific data folders
    DIR_ILP: str = os.path.join(DATA_DIR, "Data_InductiveLinkPrediction")
    DIR_TLP: str = os.path.join(DATA_DIR, "Data_TransductiveLinkPrediciton")
    
    # Hardware and model settings
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    BATCH_SIZE_TRAIN: int = 1024
    LEARNING_RATE: float = 0.01
    FANOUTS: List[int] = field(default_factory=lambda: [15, 10, 5])