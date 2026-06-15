"""PTERO type definitions and constants."""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

class AcquisitionMode(str, Enum):
    DDA = "DDA"; DIA = "DIA"; DIA_PASEF = "diaPASEF"; DDA_PASEF = "ddaPASEF"
    PRM = "PRM"; SRM_MRM = "SRM/MRM"; UNKNOWN = "unknown"

class QuantMethod(str, Enum):
    LFQ = "LFQ"; TMT = "TMT"; ITRAQ = "iTRAQ"; SILAC = "SILAC"
    DIA_QUANT = "DIA"; LABEL_FREE_DIA = "LFQ-DIA"; UNKNOWN = "unknown"

class SearchEngine(str, Enum):
    MAXQUANT = "maxquant"; DIANN = "diann"; FRAGPIPE = "fragpipe"; SAGE = "sage"
    SPECTRONAUT = "spectronaut"; GENERIC = "generic"; UNKNOWN = "unknown"

class MobilityType(str, Enum):
    TIMS_INV_K0 = "TIMS_1/K0"; CCS = "CCS"; DRIFT_TIME = "drift_time"; NONE = "none"

class NormMethod(str, Enum):
    NONE = "none"; MEDIAN = "median"; QUANTILE = "quantile"; TOTAL = "total"

class ImputeMethod(str, Enum):
    NONE = "none"; MIN_PROB = "minprob"; KNN = "knn"; HYBRID = "hybrid"; ZERO = "zero"

class MissingType(str, Enum):
    MNAR = "MNAR"; MAR = "MAR"; UNKNOWN = "unknown"

@dataclass
class InstrumentInfo:
    name: str = "unknown"; vendor: str = "unknown"; model: str = "unknown"
    has_ion_mobility: bool = False; mobility_type: MobilityType = MobilityType.NONE

@dataclass
class ExperimentMetadata:
    name: str = "Untitled Experiment"; description: str = ""; organism: str = ""
    instrument: InstrumentInfo = field(default_factory=InstrumentInfo)
    acquisition_mode: AcquisitionMode = AcquisitionMode.UNKNOWN
    quant_method: QuantMethod = QuantMethod.UNKNOWN
    search_engine: SearchEngine = SearchEngine.UNKNOWN
    fasta_file: str = ""; enzyme: str = "trypsin"
    fixed_modifications: list[str] = field(default_factory=lambda: ["Carbamidomethyl (C)"])
    variable_modifications: list[str] = field(default_factory=lambda: ["Oxidation (M)"])
    fdr_threshold: float = 0.01; ptero_version: str = "0.1.0"

LAYER_RAW = "raw"; LAYER_NORMALIZED = "normalized"; LAYER_IMPUTED = "imputed"
LAYER_LOG2 = "log2"; LAYER_BATCH_CORRECTED = "batch_corrected"

# Data scale thresholds for rendering backend selection
SMALL_DATA_THRESHOLD = 100_000
MEDIUM_DATA_THRESHOLD = 10_000_000
