from plugins.plugin_model import ClosedPlugin
import pickle


class ShaftlessPlugin(ClosedPlugin):

    def shaftless_an(self, corrected_image, result_organizer, further_correction_file, output_path, correction_data):
        print("Running shaftless AM")
        plugin_path = r"D:\Andrea\OneDrive - Alma Mater Studiorum Università di Bologna\Work\Progetti\SW\_py\Esis-gigapixel\plugins\closed\plugin2\bin\shaftless.exe"

        # Save image as tif
        image_path = result_organizer.save_image(corrected_image, format="tif")

        return self.execute_cli(plugin_path, f"-i {image_path} -m AM -c {further_correction_file} -o {output_path}")

        # with open('correction_file', 'rb') as f:
        #     shaftless_correction = pickle.load(f)
        # correction_data.set_custom_property("SHAFTLESS", shaftless_correction)

        # retexecute_cli(
        #     "shaftless.exe", f"-i {image_path}", f"-m AM-c {self.bin_path}"
        # )

    def shaftless_de(self, corrected_image, result_organizer, further_correction_file, output_path, correction_data):
        print("Running shaftless DM")
        plugin_path = r"D:\Andrea\OneDrive - Alma Mater Studiorum Università di Bologna\Work\Progetti\SW\_py\Esis-gigapixel\plugins\closed\plugin2\bin\shaftless.exe"

        # Save image as tif
        image_path = result_organizer.save_image(corrected_image, format="tif")

        return self.execute_cli(plugin_path, f"-i {image_path} -m DM -c {further_correction_file} -o {output_path}")
        # return self.execute_cli(
        #     "shaftless.exe",
        #     f"-i {image_path}",
        #     f"-m DM-c {self.bin_path}",
        #     f"-o .\output",
        # )


plugin = ShaftlessPlugin()
