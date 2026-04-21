import smtplib
from email.mime.text import MIMEText

def send_email(subject, body, to_email):
    from_email = "your_email@gmail.com"
    password = "your_password"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(from_email, password)
    server.send_message(msg)
    server.quit()


if __name__ == "__main__":
    send_email(
        "Risk Alert",
        "High supply chain risk detected!",
        "receiver@gmail.com"
    )