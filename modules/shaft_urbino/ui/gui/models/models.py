from dataclasses import dataclass, field
from typing import List, Optional

from core.common.equipment.equipment import Equipment


@dataclass
class CalibrationDistortion:
    model: str  # poly3, poly5, ptlens
    focal: float
    real_focal: Optional[float] = None
    k1: Optional[float] = None
    k2: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None


@dataclass
class CalibrationTca:
    model: str  # linear, poly3
    focal: float
    kr: Optional[float] = None
    kb: Optional[float] = None
    vr: Optional[float] = None
    vb: Optional[float] = None
    cr: Optional[float] = None
    cb: Optional[float] = None
    br: Optional[float] = None
    bb: Optional[float] = None


@dataclass
class CalibrationVignetting:
    model: str  # pa
    focal: float
    aperture: float
    distance: float
    k1: float
    k2: float
    k3: float


@dataclass
class Camera:
    maker: str
    model: str
    mount: str
    cropfactor: float


@dataclass
class Lens:
    maker: str
    model: str
    mount: str
    cropfactor: float


@dataclass
# Empty calibration data
class Calibration:
    camera: Camera = None
    lens: Lens = None
    distortion: CalibrationDistortion = None
    # tcas: List[CalibrationTca] = None
    vignetting: CalibrationVignetting = None


@dataclass
class CalibrationData:
    equipment: Equipment
    calibration: Calibration
