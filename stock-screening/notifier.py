import requests
import os

def send_telegram_alert(message):
    """
    Sends a text message to your personal Telegram chat.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return "Error: Telegram credentials (Bot Token or Chat ID) are missing."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
        # We removed "parse_mode": "Markdown" because LLM text breaks Telegram's strict parser!
    }

    try:
        response = requests.post(url, json=payload)

        # If Telegram rejects it, this will tell us exactly why (e.g., "chat not found")
        if response.status_code != 200:
            return f"Telegram API Error: {response.text}"

        return "Success"
    except Exception as e:
        return f"Failed to send Telegram message: {e}"