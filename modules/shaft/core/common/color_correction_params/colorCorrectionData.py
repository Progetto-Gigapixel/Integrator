import json
from argparse import BooleanOptionalAction
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from config.config import AnalysisSteps

from .model import *


@dataclass
class ColorCorrectionData:
    """Data class for storing color correction parameters."""
    exposure_correction: Optional[ExposureCorrectionData] = None
    white_balance_correction: Optional[WhiteBalanceData] = None
    flat_fielding_correction_path: Optional[FlatFieldingData] = None
    denoising: Optional[DenoisingData] = None
    ccm: Optional[CCM] = None
    ridge_ccm: Optional[RidgeCCM] = None
    finlayson_ccm: Optional[FinlaysonCCM] = None
    wls_correction: Optional[WlsData] = None
    polynomial_correction: Optional[PolynomialFitting] = None
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    shaftless_correction: Optional[ShaftlessData] = None
    rawtherapee_correction: Optional[RawtherapeeData] = None

    def to_dict(self):
        data = asdict(self)
        # Unpack custom_properties into the main dict
        data.update(self.custom_properties)
        # Remove the custom_properties key to avoid duplication
        data.pop("custom_properties", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        # Extract known fields
        known_fields = {
            field.name
            for field in cls.__dataclass_fields__.values()
            if field.name != "custom_properties"
        }
        init_data = {k: v for k, v in data.items() if k in known_fields}
        custom_data = {k: v for k, v in data.items() if k not in known_fields}

        instance = cls()

        # Initialize standard fields
        for key, value in init_data.items():
            if key == AnalysisSteps.EXPOSURE and value is not None:
                instance.exposure_correction = ExposureCorrectionData(**value)
            elif key == AnalysisSteps.WHITE_BALANCE and value is not None:
                instance.white_balance_correction = WhiteBalanceData(**value)
            elif key == AnalysisSteps.FLAT_FIELDING and value is not None:
                instance.flat_fielding_correction = FlatFieldingData(**value)
            elif key == AnalysisSteps.DENOISING and value is not None:
                instance.denoising = DenoisingData(**value)
            elif key == AnalysisSteps.CCM and value is not None:
                instance.ccm = CCM(**value)
            elif key == AnalysisSteps.RIDGE_CCM and value is not None:
                instance.ridge_ccm = RidgeCCM(**value)
            elif key == AnalysisSteps.FINLAYSON_CCM and value is not None:
                instance.finlayson_ccm = FinlaysonCCM(**value)
            elif key == AnalysisSteps.WLS and value is not None:
                instance.wls_correction = (
                    WlsData(**value) if isinstance(value, dict) else value
                )
            elif (
                key == AnalysisSteps.POLYNOMIAL_FITTING_CORRECTOR and value is not None
            ):
                instance.polynomial_correction = PolynomialFitting(**value)
            elif key == AnalysisSteps.SHAFTLESS and value is not None:
                instance.shaftless_correction = (
                    ShaftlessData(**value) if isinstance(value, dict) else value
                )
            elif key == AnalysisSteps.RAWTHERAPEE and value is not None:
                sharpen = value.get("raw_therapee_sharpen")
                light_balance = value.get("raw_therapee_light_balance")
                rt_data = RawtherapeeData(sharpen,light_balance)
                instance.rawtherapee_correction = rt_data


        # Assign custom properties
        instance.custom_properties = custom_data

        return instance

    def set_norm_exp_correction(self, expected_minimum, expected_range, measured_range, saturation_correction, v_measured_black):
        self.exposure_correction = ExposureCorrectionData(
            expected_minimum, expected_range, measured_range, saturation_correction, v_measured_black
        )

    def get_exposure_correction(self):
        return (
            self.exposure_correction.expected_minimum,
            self.exposure_correction.expected_range,
            self.exposure_correction.measured_range,
            self.exposure_correction.saturation_correction,
            self.exposure_correction.v_measured_black,
        )

    def set_white_balance_correction(self, src_white_point, dst_white_point):
        self.white_balance_correction = WhiteBalanceData(
            src_white_point, dst_white_point
        )

    def get_white_balance_correction(self):
        return (
            self.white_balance_correction.src_white_point,
            self.white_balance_correction.dst_white_point,
        )

    def set_flat_fielding_correction(self, flat_fielding_correction_path):
        self.flat_fielding_correction = flat_fielding_correction_path

    def get_flat_fielding_correction(self):
        return self.flat_fielding_correction_path

    def set_denoising(self, bm3d_strength):
        self.denoising = DenoisingData(bm3d_strength=bm3d_strength)

    def get_denoising(self):
        return self.denoising.bm3d_strength


    def set_ccm(self, t):
        self.ccm = CCM(t=t)

    def get_ccm(self):
        return self.ccm.t

    def set_ridge_ccm(self, ccm, measured_values):
        self.ridge_ccm = RidgeCCM(ccm=ccm, measured_values=measured_values)

    def get_ridge_ccm(self):
        return (
            self.ridge_ccm.ccm,
            self.ridge_ccm.measured_values,
        )

    def set_finlayson_ccm(self, measured_values):
        self.finlayson_ccm = FinlaysonCCM(measured_values=measured_values)

    def get_finlayson_ccm(self):
        return self.finlayson_ccm.measured_values

    def set_wls_correction(self, measured_values):
        self.wls_correction = WlsData(measured_values=measured_values)

    def get_wls_correction(self):
        return self.wls_correction.measured_values

    def set_polynomial_correction(self, degree, measured_values):
        self.polynomial_correction = PolynomialFitting(
            degree=degree, measured_values=measured_values
        )

    def get_polynomial_correction(self):
        return (
            self.polynomial_correction.degree,
            self.polynomial_correction.measured_values,
        )

    def set_shaftless_correction(self, measured_values):
        self.shaftless_correction = ShaftlessData(measured_values=measured_values)

    def get_shaftless_correction(self):
        return self.shaftless_correction.measured_values

    def set_rawtherapee_correction(self, sharpen, light_balance):
        self.rawtherapee_correction = RawtherapeeData(sharpen,light_balance)

    def get_rawtherapee_correction(self):
        return self.rawtherapee_correction

    def set_custom_property(self, key: str, value: Any):
        self.custom_properties[key] = value

    def get_custom_property(self, key: str) -> Any:
        return self.custom_properties.get(key, None)

    def list_custom_properties(self):
        return list(self.custom_properties.keys())



