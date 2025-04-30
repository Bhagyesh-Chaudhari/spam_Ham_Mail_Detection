from flask import Flask, render_template
import imaplib
import email
import joblib
import re
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load(r'D:\WORKSPACE\spam_Ham_Mail_Detection\spam_ham_mail_detection\spam_classifier_model.joblib')
vectorizer = joblib.load(r'D:\WORKSPACE\spam_Ham_Mail_Detection\spam_ham_mail_detection\spam_vectorizer.joblib')

# Preprocessing function (same as during training)
def preprocess_text(text):
    text = str(text)
    text = re.sub(r'(?i)subject:', '', text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to fetch emails from Gmail
def fetch_emails(email_user, email_pass, n=10):
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(email_user, email_pass)
    mail.select('inbox')
    result, data = mail.search(None, 'ALL')
    mail_ids = data[0].split()
    emails = []
    for i in mail_ids[-n:]:
        result, msg_data = mail.fetch(i, '(RFC822)')
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        subject = msg['subject']
        if subject:
            clean_subject = preprocess_text(subject)
            vect = vectorizer.transform([clean_subject])
            label = model.predict(vect)[0]
            emails.append({'subject': subject, 'label': label})
    mail.logout()
    return emails[::-1]  # Most recent first

app = Flask(__name__)

@app.route('/')
def index():
    # Replace with your email and app password
    EMAIL = 'bhagyeshchaudhari71@gmail.com'
    PASSWORD = 'kzpt iyye gwue muzx'
    try:
        emails = fetch_emails(EMAIL, PASSWORD, n=10)
    except Exception as e:
        emails = []
        print("Error fetching emails:", e)
    # Render the template with the emails
    return render_template('index.html', emails=emails)

if __name__ == '__main__':
    app.run(debug=True)
