class ComparisonStep:
    def __init__(self):
        self.step_applied = False
        self.delta_e_mean = None
        self.delta_e_sd = None
        self.delta_e_max = None
        self.delta_e_min = None
        self.delta_e_weighted_average = None
        self.patches = []

    def add_patch(self, patch):
        self.patches.append(patch)


    def to_dict(self):
        return {
            "STEP_APPLIED": self.step_applied,
            "delta_e_mean": self.delta_e_mean,
            "delta_e_sd": self.delta_e_sd,
            "delta_e_max": self.delta_e_max,
            "delta_e_min": self.delta_e_min,
            "delta_e_weighted_average": self.delta_e_weighted_average,
            "patches": self.patches,
        }
