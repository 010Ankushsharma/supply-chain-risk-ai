import requests
import json

WEBHOOK_URL = "YOUR_SLACK_WEBHOOK_URL"

def send_alert(message):
    payload = {"text": message}

    requests.post(WEBHOOK_URL, data=json.dumps(payload))


if __name__ == "__main__":
    send_alert("🚨 Test Alert from Supply Chain AI")