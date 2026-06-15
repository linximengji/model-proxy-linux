"""Route loading and env resolution. No external deps beyond yaml + os."""
import os

CONFIG_PATH = r"D:\ClaudeProjects\proxy\litellm_config.yaml"
DOTENV_PATH = r"D:\ClaudeProjects\proxy\.env"

ENV_KEY_MAP = {
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "QWEN_MAAS_API_KEY",
    "openai": "DASHSCOPE_API_KEY",
    "doubao": "DOUBAO_API_KEY",
}


def load_dotenv(path=DOTENV_PATH):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_env(value):
    if isinstance(value, str) and value.startswith("os.environ/"):
        var_name = value.split("/", 1)[1]
        return os.environ.get(var_name, value)
    return value


def load_routes(config_path=CONFIG_PATH):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    routes = {}
    for m in cfg["model_list"]:
        name = m["model_name"]
        params = m["litellm_params"]
        p = params.copy()
        model = p["model"]
        if model.startswith("openai/"):
            p["model"] = model.split("/", 1)[1]
            p["provider"] = "openai"
        elif model.startswith("deepseek/"):
            p["model"] = model.split("/", 1)[1]
            p["provider"] = "deepseek"
        elif model.startswith("anthropic/"):
            p["model"] = model.split("/", 1)[1]
            p["provider"] = "anthropic"
        else:
            continue
        if p["provider"] == "deepseek" and "api_base" not in p:
            p["api_base"] = "https://api.deepseek.com/anthropic/v1/messages"
        p["api_key"] = resolve_env(p.get("api_key", ""))
        env_var = ENV_KEY_MAP.get(p["provider"])
        if env_var and os.environ.get(env_var) and p["api_key"].startswith("os.environ/"):
            p["api_key"] = os.environ[env_var]
        routes[name] = p
        fb = m.get("fallback")
        if fb:
            p["fallback"] = fb
        qb = params.get("quota_backup")
        if qb:
            p["quota_backup"] = qb
    return routes
