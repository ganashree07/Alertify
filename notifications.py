# notifications.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class NotificationSystem:
    def __init__(self, email_config):
        self.sender_email = email_config['user@gmail.com']
        self.sender_password = email_config['password']
        self.police_email = email_config['police@gmail.com']
        self.ambulance_email = email_config['ambulance@gmail.com']

    def send_email(self, to_email, subject, body):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            print(f"Alert email sent successfully to {to_email}")
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

    def notify_police(self, location, severity, incident_details, timestamp):
        subject = f"Traffic Accident Alert - Severity Level: {severity}"
        body = f"""
TRAFFIC ACCIDENT DETECTED

Location: {location}
Time: {timestamp}
Severity Level: {severity}/5

Incident Details:
{incident_details}

This is an automated alert from the Traffic Accident Detection System.
Immediate response may be required.
"""
        return self.send_email(self.police_email, subject, body)

    def notify_ambulance(self, location, severity, incident_details, timestamp):
        subject = f"URGENT: Medical Assistance Required - Severity Level: {severity}"
        body = f"""
URGENT: SEVERE TRAFFIC ACCIDENT DETECTED

Location: {location}
Time: {timestamp}
Severity Level: {severity}/5

Incident Details:
{incident_details}

IMMEDIATE MEDICAL ASSISTANCE REQUIRED
This is an automated alert from the Traffic Accident Detection System.
"""
        return self.send_email(self.ambulance_email, subject, body)
