from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExposureCorrectionData:
    expected_minimum: Optional[float] = (None,)
    expected_range: Optional[float] = (None,)
    measured_range: Optional[float] = (None,)
    saturation_correction: Optional[float] = (None,)
    v_measured_black: Optional[float] = (None,)


@dataclass
class WhiteBalanceData:
    src_white_point: Optional[List[float]] = None
    dst_white_point: Optional[List[float]] = None


@dataclass
class FlatFieldingData:
    gain_matrix: Optional[List[float]] = None


@dataclass
class DenoisingData:
    bm3d_strength: Optional[float] = None


@dataclass
class CCM:
    t: Optional[List[List[float]]] = None


@dataclass
class RidgeCCM:
    ccm: Optional[List[List[float]]] = None
    measured_values: Optional[List[float]] = None


@dataclass
class FinlaysonCCM:
    measured_values: Optional[List[float]] = None


@dataclass
class PolynomialFitting:
    degree: Optional[int] = None
    measured_values: Optional[List[float]] = None


@dataclass
class WlsData:
    measured_values: Optional[List[float]] = None

@dataclass
class ShaftlessData:
    measured_values: Optional[List[float]] = None

@dataclass
class RawtherapeeData:
    raw_therapee_sharpen: Optional[bool] = None
    raw_therapee_light_balance: Optional[bool] = None

