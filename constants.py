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

# MySQL DATABASE CONFIGURATION
MEMORY_DB_CONFIG = {
    "DB_USER": getenv("DB_USER"),
    "DB_PASSWORD": getenv("DB_PASSWORD"),
    "HOST": getenv("HOST"),
    "MEMORY_DB_NAME": getenv("MEMORY_DB_NAME"),
}

MEMORY_MYSQL_URL = (
    f"mysql+pymysql://{MEMORY_DB_CONFIG['DB_USER']}:"
    f"{MEMORY_DB_CONFIG['DB_PASSWORD']}@"
    f"{MEMORY_DB_CONFIG['HOST']}/{MEMORY_DB_CONFIG['MEMORY_DB_NAME']}"
)
