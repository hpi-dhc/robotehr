import os

BASE_PATH = os.getenv("ROBOTEHR_BASE_PATH") or input("Base Path:")
DB_URI = os.getenv("ROBOTEHR_DB_URI") or input("Database URI:")
WEBHOOK_URL = os.getenv("ROBOTEHR_WEBHOOK_URL") or input("Webhook URL:")
