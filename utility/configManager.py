import json
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path='config.json'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load the configuration from JSON file"""
        try:
            with open(self.config_path) as f:
                print("Config file loaded" + str(self.config_path))
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default empty config if file doesn't exist or is invalid
            return {
                "app": {},
                "paths": {},
                "venvs": {},
                "imagematcher": {}
            }
    
    def save_config(self):
        """Save the current configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def update_value(self, section, key, value):
        """Update a configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
    
    def get_value(self, section, key, default=None):
        """Get a configuration value"""
        try:
            return self.config[section][key]
        except KeyError:
            return default
