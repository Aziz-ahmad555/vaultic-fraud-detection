# utils.py
import os
import smtplib
import hashlib
from email.mime.text import MIMEText
from joblib import dump, load


def make_token(user_id):
    return hashlib.sha256(f"{user_id}-{os.urandom(16)}".encode()).hexdigest()


def send_email_alert(to_email, subject, body):
    """
    Send a fraud-alert email over SMTP.

    Reads config from environment variables:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_DEV_MODE

    If MAIL_DEV_MODE=1 (or no SMTP_USER is configured), the email is just
    printed to the console instead of sent — so the alert flow can still be
    demoed end-to-end without real SMTP credentials.

    Returns (success: bool, detail: str).
    """
    dev_mode = os.environ.get("MAIL_DEV_MODE", "1") == "1"
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if dev_mode or not smtp_user or not smtp_pass or not to_email:
        print(f"[DEV MAIL] To: {to_email} | Subject: {subject}\n{body}\n")
        return True, "dev-mode (printed to console, not actually sent)"

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        return True, "sent"
    except Exception as e:
        # Never let a mail failure break the transaction flow.
        print(f"[MAIL ERROR] {e}")
        return False, str(e)


def send_sms_alert(body):
    """
    Send a fraud-alert SMS via Twilio.

    Reads config from environment variables:
      TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER,
      ALERT_PHONE_NUMBER

    If any of these are missing, the SMS is just printed to the console
    instead of sent -- same dev-mode-fallback pattern as send_email_alert,
    so the alert flow can be demoed without real Twilio credentials.

    NOTE: on a Twilio trial account, SMS can only be sent to phone numbers
    verified in the Twilio console (Phone Numbers -> Manage -> Verified
    Caller IDs). ALERT_PHONE_NUMBER should be one of those verified numbers.

    Returns (success: bool, detail: str).
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_PHONE_NUMBER")
    to_number = os.environ.get("ALERT_PHONE_NUMBER")

    if not all([account_sid, auth_token, from_number, to_number]):
        print(f"[DEV SMS] To: {to_number} | {body}\n")
        return True, "dev-mode (printed to console, Twilio credentials not fully set)"

    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body,
            from_=from_number,
            to=to_number,
        )
        return True, f"sent (sid={message.sid})"
    except Exception as e:
        # Never let an SMS failure break the transaction flow.
        print(f"[SMS ERROR] {e}")
        return False, str(e)
