from plugins.plugin_model import ClosedPlugin


class Echo(ClosedPlugin):
    def echo_an(self, param1, param2, correction_data):
        print("Running echo an")

        correction_data.set_custom_property("echo", "echo")

        return self.execute_cli("echo", f"param1: {param1}, param2: {param2}")

    def echo_de(self, param1, param2, correction_data):
        print("Running echo de")

        correction_data.set_custom_property("echo", "echo")

        return self.execute_cli("echo", f"param1: {param1}, param2: {param2}")


plugin = Echo()
