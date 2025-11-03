from core.common.exceptions.startup_checker_exceptions import (
    LensfunMissingCamera,
    LensfunMissingData,
    LensfunMissingLens,
    StartupCheckerException,
)
from core.utils.core_utils import get_lensfun_db
from locales.localization import _
from log.logger import logger


class StartupChecker:
    def __init__(self, equipment):
        """
        Checks the startup conditions for the application.

        :param equipment: The equipment used to capture the image
        """
        self.equipment = equipment

    # Start the initial check
    def start_check(self):
        """Starts the initial check"""
        # self._lensfun_check()
        pass

    # Check the lensfun database
    def _lensfun_check(self):
        """Check if camera and lens are in the lensfun database"""
        try:
            db = get_lensfun_db()
            camera_make = self.equipment.camera_make
            camera_model = self.equipment.camera_model
            lens_make = self.equipment.lens_make
            lens_model = self.equipment.lens_model

            cameras = db.find_cameras(camera_make, camera_model)
            # if not cameras:
            #     raise LensfunMissingCamera(_("No camera found in the lensfun database."))
            camera = cameras[0]
            logger.info(f"Camera found: {camera}")

            lenses = db.find_lenses(camera, lens_make, lens_model)
            lens = lenses[0]
            logger.info(f"Lense found: {lens}")
            # if not lenses:
            #     raise LensfunMissingLens(_("No lens found in the lensfun database."))
        except LensfunMissingData as LMD:
            logger.error(_("Failed to check lensfun database: %s"), LMD)
            raise LMD
