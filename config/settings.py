import yaml
import os

def get_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def get_openai_api_key():
    # Looks for .env or system env var
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY"):
                    return line.strip().split("=", 1)[1]
    return os.environ.get("OPENAI_API_KEY", "")
