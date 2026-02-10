import os
import yaml
import re
from dotenv import load_dotenv


class ConfigLoader:
    def __init__(self, config_path="config.yaml", env_path=".env"):
        load_dotenv(env_path)

        with open(config_path, "r", encoding="utf-8") as file:
            loaded_config = yaml.safe_load(file)

        # Resolve ${VAR} placeholders across the entire config tree up front
        self.config = self._resolve_tree(loaded_config)
        self.user_constants = {}

    def _resolve_env_vars(self, value):
        """Resolve ${VAR_NAME} references to environment variables."""
        if isinstance(value, str):
            # Replace ${VAR_NAME} with environment variable values
            def replace_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))

            return re.sub(r'\$\{([^}]+)\}', replace_var, value)
        return value

    def _resolve_tree(self, node):
        """Recursively resolve env placeholders in nested config structures."""
        if isinstance(node, dict):
            return {k: self._resolve_tree(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._resolve_tree(item) for item in node]
        return self._resolve_env_vars(node)

    def load_global_env_vars(self):
        """Load global environment variables from config.yaml if not already set in .env."""
        env_config = self.config.get("env", {})
        for key, value in env_config.items():
            if key not in os.environ:
                resolved_value = self._resolve_env_vars(str(value))
                # If the value still contains an unresolved placeholder like ${VAR}, skip setting
                if isinstance(resolved_value, str) and "${" in resolved_value and "}" in resolved_value:
                    continue
                os.environ[key] = resolved_value

    def load_user_config(self, user: str, broker: str):
        users = self.config.get("users", {})
        if user not in users:
            raise ValueError(f"User '{user}' not found in configuration.")
        user_config = users[user]
        if broker not in user_config:
            raise ValueError(f"Broker '{broker}' configuration for user '{user}' not found.")
        broker_config = user_config[broker]

        env_vars = broker_config.get("env", {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)

        self.user_constants = broker_config.get("constant", {})

    def get_user_constant(self, key):
        return self.user_constants.get(key)

    def get_env_var(self, key):
        return os.environ.get(key)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    config_loader.load_global_env_vars()

    config_loader.load_user_config("junting", "fugle")

    print("FINLAB_API_TOKEN:", os.environ.get("FINLAB_API_TOKEN"))
    print("FUGLE_ACCOUNT:", os.environ.get("FUGLE_ACCOUNT"))
    print("rebalance_safety_weight (from constant):", config_loader.get_user_constant("rebalance_safety_weight"))
    print("strategy_class_name (from constant):", config_loader.get_user_constant("strategy_class_name"))
