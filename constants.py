import os
from os.path import dirname, abspath, join
from urllib.parse import quote_plus


def getenv(env_var, fallback=""):
    return os.getenv(env_var, fallback).strip()


# Configure Environment
DEV_ENV = "dev"
STAGING_ENV = "staging"
PROD_ENV = "production"

GOOGLE_ACCESS_TOKEN = getenv("GOOGLE_ACCESS_TOKEN")

current_env = getenv("ENV", DEV_ENV)
if current_env not in [DEV_ENV, STAGING_ENV, PROD_ENV]:
    current_env = DEV_ENV

ENV = current_env

PROJECT_ID = getenv("PROJECT_ID", "typoapp-442017")
TOOLBOX_URL = getenv("TOOLBOX_URL", "https://toolbox-aua232uyqa-uc.a.run.app")
ORG_ID = int(getenv("ORG_ID", "5"))
DEFAULT_VERTEX_AI_MODEL_NAME = getenv(
    "DEFAULT_VERTEX_AI_MODEL_NAME", "gemini-2.5-flash"
)
