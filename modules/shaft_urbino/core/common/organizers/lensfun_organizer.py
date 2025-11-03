import json
import os
from pathlib import Path

from lxml import etree as ET

from log.logger import logger
from utils.utils import read_config, find_project_root


class LensfunOrganizer:
    def __init__(
        self, equipment, camera_info, lens_info, geometric_data, vignetting_data
    ):
        """
        Initializes the Lensfun organizer for creating the custom lensfun database.

        :param equipment: The equipment used to capture the image
        :param camera_info: The camera information
        :param lens_info: The lens information
        :param geometric_data: The geometric data for the lens
        :param vignetting_data: The vignetting data for the lens
        """
        self.equipment = equipment
        self.camera_info = camera_info
        self.lens_info = lens_info
        self.geometric_data = geometric_data
        self.vignetting_data = vignetting_data

    # Create the XML file for the custom lensfun database
    def create_xml(self):
        logger.info("Creating XML file for the custom lensfun database")
        config = read_config()

        # Path to the custom lensfun database
        lensfun_path = os.path.abspath(config.get("directories", "lensfun_path"))
        lensfun_path.mkdir(
            parents=True, exist_ok=True
        )  # Create directory if it doesn't exist
        file_name = f"{self.lens_info.maker}_{self.lens_info.model}.xml"
        file_name = file_name.replace("/", "_")  # remove / from the file name
        file_path = lensfun_path / file_name

        lens_database = ET.Element("lensdatabase", version="1")

        # Create the lens section
        lens = ET.SubElement(lens_database, "lens")
        ET.SubElement(lens, "maker").text = self.lens_info.maker
        ET.SubElement(lens, "model").text = self.lens_info.model
        ET.SubElement(lens, "mount").text = self.lens_info.mount
        ET.SubElement(lens, "cropfactor").text = str(self.lens_info.cropfactor)

        # Calibration section for the lens
        calibration = ET.SubElement(lens, "calibration")

        distortion = ET.SubElement(
            calibration, "distortion", model=self.geometric_data.model
        )
        # Assegna attributi basati sul modello di distorsione
        if self.geometric_data.model in ["poly3", "poly5"]:
            distortion.set("focal", str(self.geometric_data.focal))
            distortion.set("k1", str(self.geometric_data.k1))
            if self.geometric_data.model == "poly5":
                distortion.set("k2", str(self.geometric_data.k2))
        elif self.geometric_data.model == "ptlens":
            distortion.set("focal", str(self.geometric_data.focal))
            distortion.set("a", str(self.geometric_data.a))
            distortion.set("b", str(self.geometric_data.b))
            distortion.set("c", str(self.geometric_data.c))

        # Vignetting section for the lens
        vignetting = ET.SubElement(
            calibration, "vignetting", model=self.vignetting_data.model
        )
        vignetting.set("focal", str(self.vignetting_data.focal))
        vignetting.set("aperture", str(self.vignetting_data.aperture))
        vignetting.set("distance", str(self.vignetting_data.distance))
        vignetting.set("k1", str(self.vignetting_data.k1))
        vignetting.set("k2", str(self.vignetting_data.k2))
        vignetting.set("k3", str(self.vignetting_data.k3))

        # Create the camera section
        camera = ET.SubElement(lens_database, "camera")
        ET.SubElement(camera, "maker").text = self.camera_info.maker
        ET.SubElement(camera, "model").text = self.camera_info.model
        ET.SubElement(camera, "mount").text = self.camera_info.mount
        ET.SubElement(camera, "cropfactor").text = str(self.camera_info.cropfactor)

        logger.info(f"Path to save the XML file: {file_path}")

        # Convert the XML tree to a string and save it
        tree = ET.ElementTree(lens_database)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

        return f"XML file saved at {file_path}"
