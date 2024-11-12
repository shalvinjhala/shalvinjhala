- 👋 Hi, I’m @shalvinjhala
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
shalvinjhala/shalvinjhala is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import speech_recognition as sr
import pyttsx3

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen to user
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        print("Recognizing...")
        try:
            command = recognizer.recognize_google(audio)
            print(f"User said: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, I'm having trouble connecting to the service.")
            return None

# Example usage
while True:
    command = listen()
    if command:
        speak(f"You said: {command}")
        # Add more conditions to execute specific commands based on user input

from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(_name_)

# Function to connect to the database
def connect_db():
    return sqlite3.connect('user_data.db')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Connect to the database and check credentials
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return redirect(url_for('dashboard'))
        else:
            return "Invalid login credentials"
    
    return render_template('login.html')

# Route for dashboard page
@app.route('/dashboard')
def dashboard():
    return "Welcome to your Dashboard"

if _name_ == '_main_':
    app.run(debug=True)


import requests

def get_weather(city):
    API_KEY = 'your_openweather_api_key'
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={API_KEY}"
    
    response = requests.get(complete_url)
    data = response.json()
    
    if data["cod"] != "404":
        main_data = data["main"]
        weather_data = data["weather"][0]
        temp = main_data["temp"] - 273.15  # Convert from Kelvin to Celsius
        description = weather_data["description"]
        return f"The temperature in {city} is {temp:.2f}°C with {description}."
    else:
        return "City not found."

# Example usage
city = "New York"
print(get_weather(city))

import sqlite3
from datetime import datetime

# Function to log user activities
def log_activity(user_id, activity):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO activity_log (user_id, activity, timestamp) VALUES (?, ?, ?)", (user_id, activity, timestamp))
    conn.commit()
    conn.close()

# Function to file a complaint
def file_complaint(user_id, complaint_details):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO complaints (user_id, complaint_details, timestamp) VALUES (?, ?, ?)", (user_id, complaint_details, timestamp))
    conn.commit()
    conn.close()

# Example usage
log_activity(1, "User accessed restricted area")
file_complaint(1, "Inappropriate content detected")

import sqlite3

def generate_report():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM activity_log")
    logs = cursor.fetchall()
    conn.close()
    
    for log in logs:
        print(f"User ID: {log[0]}, Activity: {log[1]}, Timestamp: {log[2]}")

# Example usage
generate_report()


from twilio.rest import Client

# Function to send message via WhatsApp
def send_whatsapp_message(to_number, message):
    account_sid = 'your_account_sid'
    auth_token = 'your_auth_token'
    client = Client(account_sid, auth_token)
    
    message = client.messages.create(
        body=message,
        from_='whatsapp:+14155238886',
        to=f'whatsapp:{to_number}'
    )
    print(f"Message sent to {to_number}")
    
# Example usage
send_whatsapp_message("+1234567890", "Hello, this is a test message.")


import tweepy

# Function to post a tweet
def post_tweet(message):
    consumer_key = 'your_consumer_key'
    consumer_secret = 'your_consumer_secret'
    access_token = 'your_access_token'
    access_token_secret = 'your_access_token_secret'
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    api = tweepy.API(auth)
    api.update_status(message)
    print(f"Tweet sent: {message}")

# Example usage
post_tweet("This is an automated tweet from my AI assistant!")


import os
import subprocess

def execute_command(command):
    try:
        output = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output.stdout.decode()
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"

# Example usage
command_output = execute_command('ls')
print(f"Command output: {command_output}")



from PIL import Image, ImageDraw, ImageFont

def create_image_with_text(text):
    img = Image.new('RGB', (500, 500), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    d.text((10,10), text, font=font, fill=(255, 255, 0))
    
    img.save('generated_image.png')
    print("Image created successfully!")
    
# Example usage
create_image_with_text("Hello, AI Assistant!")


import cv2
import numpy as np

def create_video_from_images(image_folder, output_video_file):
    images = [f"{image_folder}/image_{i}.png" for i in range(1, 6)]  # example image filenames
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, 1, (width, height))
    
    for image in images:
        video.write(cv2.imread(image))
    
    video.release()
    print(f"Video {output_video_file} created successfully!")

# Example usage
create_video_from_images('images_folder', 'output_video.mp4')


import json

def change_preference(preference, value):
    settings = load_settings()
    settings[preference] = value
    save_settings(settings)
    print(f"{preference} has been updated to {value}")

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        settings = {"language": "English", "theme": "Light"}
    return settings

def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

# Example usage
change_preference("theme", "Dark")


import shutil
import os

def backup_data(source_folder, backup_folder):
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    shutil.copytree(source_folder, backup_folder)
    print(f"Data backed up from {source_folder} to {backup_folder}")

def restore_data(backup_folder, restore_folder):
    if os.path.exists(restore_folder):
        shutil.rmtree(restore_folder)
    shutil.copytree(backup_folder, restore_folder)
    print(f"Data restored from {backup_folder} to {restore_folder}")

# Example usage
backup_data('user_data', 'backup_data')
restore_data('backup_data', 'restored_data')


import speech_recognition as sr

def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"User said: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return ""

# Example usage
command = listen_to_user()


import pyttsx3

def speak_response(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

# Example usage
speak_response("Hello, how can I assist you today?")



import sqlite3

# Connect to the database
def connect_to_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    return conn, cursor

# Create a table to store user data (if not exists)
def create_table():
    conn, cursor = connect_to_db()
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_data (
                        id INTEGER PRIMARY KEY, 
                        name TEXT,
                        email TEXT)''')
    conn.commit()
    conn.close()

# Inserting new user data
def insert_user_data(name, email):
    conn, cursor = connect_to_db()
    cursor.execute("INSERT INTO user_data (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    conn.close()

# Retrieving user data
def get_user_data():
    conn, cursor = connect_to_db()
    cursor.execute("SELECT * FROM user_data")
    users = cursor.fetchall()
    conn.close()
    return users

# Example usage
create_table()
insert_user_data("John Doe", "john@example.com")
user_data = get_user_data()
print(user_data)


import time
import datetime

def set_reminder(reminder_time, message):
    current_time = datetime.datetime.now()
    time_difference = reminder_time - current_time
    seconds_until_reminder = time_difference.total_seconds()
    
    if seconds_until_reminder > 0:
        print(f"Reminder set for {reminder_time}.")
        time.sleep(seconds_until_reminder)
        print(f"Reminder: {message}")
    else:
        print("The reminder time has already passed.")

# Example usage
reminder_time = datetime.datetime(2024, 11, 12, 9, 0)
set_reminder(reminder_time, "Meeting with client.")


import schedule

def job():
    print("This is your scheduled task!")

def schedule_task():
    schedule.every().day.at("10:00").do(job)  # Schedule the task at 10 AM every day.
    while True:
        schedule.run_pending()
        time.sleep(1)

# Example usage
schedule_task()


from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        print("Positive sentiment")
    elif sentiment < 0:
        print("Negative sentiment")
    else:
        print("Neutral sentiment")

# Example usage
analyze_sentiment("I am happy with the results!")


import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def analyze_text(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    print(pos_tags)

# Example usage
analyze_text("I love programming in Python!")


import os

def rename_file(old_name, new_name):
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        print(f"File renamed from {old_name} to {new_name}")
    else:
        print(f"The file {old_name} does not exist.")

# Example usage
rename_file("old_file.txt", "new_file.txt")


import os

def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"File {file_name} has been deleted.")
    else:
        print(f"The file {file_name} does not exist.")

# Example usage
delete_file("old_file.txt")


import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the title of the page
    title = soup.title.string
    print(f"Title of the page: {title}")
    
    # Extract all hyperlinks on the page
    links = soup.find_all('a')
    for link in links:
        print(f"Link: {link.get('href')}")

# Example usage
scrape_website('https://www.example.com')


from PIL import Image

def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path)
    resized_image = image.resize((new_width, new_height))
    resized_image.show()

# Example usage
resize_image('example_image.jpg', 800, 600)


from PIL import Image, ImageFilter

def apply_filter(image_path):
    image = Image.open(image_path)
    filtered_image = image.filter(ImageFilter.CONTOUR)
    filtered_image.show()

# Example usage
apply_filter('example_image.jpg')



import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email):
    from_email = "your_email@example.com"
    from_password = "your_password"
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, from_password)
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Example usage
send_email("Test Subject", "This is the body of the email", "recipient@example.com")



import tweepy

def post_tweet(tweet):
    consumer_key = 'your_consumer_key'
    consumer_secret = 'your_consumer_secret'
    access_token = 'your_access_token'
    access_token_secret = 'your_access_token_secret'
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    api = tweepy.API(auth)
    
    try:
        api.update_status(tweet)
        print("Tweet posted successfully!")
    except Exception as e:
        print(f"Failed to post tweet: {e}")

# Example usage
post_tweet("Hello, this is a test tweet from my AI assistant!")



import cv2

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
play_video('sample_video.mp4')



from moviepy.editor import VideoFileClip

def trim_video(input_path, output_path, start_time, end_time):
    clip = VideoFileClip(input_path)
    trimmed_clip = clip.subclip(start_time, end_time)
    trimmed_clip.write_videofile(output_path, codec='libx264')

# Example usage
trim_video('input_video.mp4', 'output_video.mp4', start_time=30, end_time=60)



from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([1, 2, 3, 4, 5])  # Dependent variable

def train_and_predict():
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict([[6]])  # Predict the value for X=6
    print(f"Predicted value: {prediction}")

# Example usage
train_and_predict()



from googletrans import Translator

def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    print(f"Translated text: {translated.text}")

# Example usage
translate_text("Hello, how are you?", "es")  # Translate to Spanish



import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_text(text):
    tokens = word_tokenize(text)
    print(f"Tokens: {tokens}")

# Example usage
tokenize_text("Hello, how are you doing today?")


import spacy

# Load the pre-trained English NLP model
nlp = spacy.load("en_core_web_sm")

def named_entity_recognition(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

# Example usage
named_entity_recognition("Apple is planning to launch new products in 2024.")



import schedule
import time

def job():
    print("This is a scheduled task!")

# Schedule the job every 5 seconds
schedule.every(5).seconds.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)


import sqlite3

# Connect to SQLite database
def connect_to_db():
    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')
    conn.commit()
    conn.close()

# Insert data into the database
def insert_data(name, age):
    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (name, age) VALUES (?, ?)", (name, age))
    conn.commit()
    conn.close()

# Fetch all users from the database
def fetch_data():
    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users")
    data = c.fetchall()
    conn.close()
    return data

# Example usage
connect_to_db()
insert_data("John Doe", 25)
print(fetch_data())



def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        print(content)

# Example usage
read_file('example.txt')



import csv

def write_to_csv(data):
    with open('example.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Example usage
data = [["Name", "Age"], ["John", 25], ["Alice", 30]]
write_to_csv(data)



import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something:")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service.")

# Example usage
recognize_speech()


from flask import Flask, render_template, request
import random

app = Flask(_name_)

def generate_response(user_input):
    responses = ["Hello!", "How can I help you?", "Good to see you!", "Have a nice day!"]
    return random.choice(responses)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = generate_response(user_input)
        return render_template("index.html", response=response)
    return render_template("index.html", response="")

if _name_ == "_main_":
    app.run(debug=True)


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
</head>
<body>
    <h1>Chat with the Bot</h1>
    <form method="post">
        <input type="text" name="user_input" placeholder="Say something" required>
        <button type="submit">Send</button>
    </form>
    <p><strong>Bot Response:</strong> {{ response }}</p>
</body>
</html>



import logging

# Set up logging configuration
logging.basicConfig(filename="app.log", level=logging.INFO)

def log_info(message):
    logging.info(f"INFO: {message}")

def log_error(message):
    logging.error(f"ERROR: {message}")

# Example usage
log_info("This is an info message.")
log_error("This is an error message.")




from PIL import Image, ImageFilter

# Open an image file
image = Image.open("example.jpg")

# Resize image
image_resized = image.resize((300, 300))
image_resized.save("resized_example.jpg")

# Rotate image
image_rotated = image.rotate(45)
image_rotated.save("rotated_example.jpg")

# Apply filter (Blur)
image_blurred = image.filter(ImageFilter.BLUR)
image_blurred.save("blurred_example.jpg")




import os
import shutil

# Rename a file
os.rename("old_name.txt", "new_name.txt")

# Move a file
shutil.move("source_folder/old_name.txt", "destination_folder/")




from flask import Flask, render_template, request, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(_name_)

# In-memory user store (for example purposes)
users_db = {}

# Signup route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)
        users_db[username] = hashed_password
        return redirect(url_for("login"))
    return render_template("signup.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = users_db.get(username)
        if hashed_password and check_password_hash(hashed_password, password):
            return "Login successful"
        return "Invalid credentials"
    return render_template("login.html")

if _name_ == "_main_":
    app.run(debug=True)


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signup</title>
</head>
<body>
    <h1>Signup</h1>
    <form method="post">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Sign Up</button>
    </form>
</body>
</html>


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
</head>
<body>
    <h1>Login</h1>
    <form method="post">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
</body>
</html>


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email):
    from_email = "your_email@example.com"
    password = "your_password"
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)

# Example usage
send_email("Test Subject", "This is a test email body", "recipient@example.com")



import requests

def get_weather(city):
    api_key = "your_openweathermap_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    if data["cod"] == 200:
        weather = data["main"]["temp"]
        description = data["weather"][0]["description"]
        print(f"Weather in {city}: {weather}°C, {description}")
    else:
        print("City not found")

# Example usage
get_weather("London")



from flask import Flask, render_template, request

app = Flask(_name_)

# Feedback route
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        user_feedback = request.form["feedback"]
        print(f"User Feedback: {user_feedback}")
        return "Thank you for your feedback!"
    return render_template("feedback.html")

if _name_ == "_main_":
    app.run(debug=True)


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Feedback</title>
</head>
<body>
    <h1>We Value Your Feedback</h1>
    <form method="post">
        <textarea name="feedback" placeholder="Enter your feedback" required></textarea>
        <button type="submit">Submit Feedback</button>
    </form>
</body>
</html>



from flask import Flask, render_template

app = Flask(_name_)

# In-memory user data (for example purposes)
users_data = [
    {"username": "JohnDoe", "email": "johndoe@example.com", "status": "Active"},
    {"username": "JaneDoe", "email": "janedoe@example.com", "status": "Inactive"}
]

@app.route("/admin")
def admin_dashboard():
    return render_template("admin_dashboard.html", users=users_data)

if _name_ == "_main_":
    app.run(debug=True)



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard</title>
</head>
<body>
    <h1>Admin Dashboard</h1>
    <table border="1">
        <thead>
            <tr>
                <th>Username</th>
                <th>Email</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {% for user in users %}
                <tr>
                    <td>{{ user.username }}</td>
                    <td>{{ user.email }}</td>
                    <td>{{ user.status }}</td>
                </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>




from flask import Flask, render_template
from flask_socketio import SocketIO, send

app = Flask(_name_)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(msg):
    print('Received message: ' + msg)
    send(msg, broadcast=True)

if _name_ == '_main_':
    socketio.run(app)




<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebSocket Example</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.min.js"></script>
    <script>
        var socket = io.connect('http://' + document.domain + ':' + location.port);

        socket.on('message', function(msg) {
            alert("New message: " + msg);
        });

        function sendMessage() {
            var msg = document.getElementById('msg').value;
            socket.send(msg);
        }
    </script>
</head>
<body>
    <h1>WebSocket Communication</h1>
    <input type="text" id="msg" placeholder="Type a message">
    <button onclick="sendMessage()">Send Message</button>
</body>
</html>




from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch()

# Index some data
doc = {
    'name': 'John Doe',
    'age': 30,
    'job': 'Software Engineer'
}
es.index(index='people', id=1, document=doc)

# Search data
query = {
    "query": {
        "match": {
            "name": "John"
        }
    }
}

response = es.search(index="people", body=query)
print(response)




import matplotlib.pyplot as plt

# Data
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 2, 5]

# Bar chart
plt.bar(categories, values)
plt.title('Category vs Value')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()




import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Line chart
plt.plot(x, y)
plt.title('X vs Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



from apscheduler.schedulers.background import BackgroundScheduler
import time

def scheduled_task():
    print("Scheduled task is running!")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_task, 'interval', seconds=10)
scheduler.start()

try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()



from flask import Flask, render_template, request

app = Flask(_name_)

users_db = {}

@app.route("/profile/<username>", methods=["GET", "POST"])
def profile(username):
    if request.method == "POST":
        email = request.form["email"]
        phone = request.form["phone"]
        users_db[username] = {"email": email, "phone": phone}
        return f"Profile updated for {username}"
    
    user_data = users_db.get(username, {"email": "", "phone": ""})
    return render_template("profile.html", user_data=user_data, username=username)

if _name_ == "_main_":
    app.run(debug=True)



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Profile</title>
</head>
<body>
    <h1>Profile of {{ username }}</h1>
    <form method="post">
        <label>Email:</label>
        <input type="email" name="email" value="{{ user_data.email }}">
        <br>
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ user_data.phone }}">
        <br>
        <button type="submit">Update Profile</button>
    </form>
</body>
</html>



import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Log a message
logging.info("This is an info message.")
logging.error("This is an error message.")



import tweepy

# Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Post a tweet
api.update_status("Hello, Twitter!")



import stripe

# Set your secret key
stripe.api_key = 'your_stripe_secret_key'

# Create a payment intent
payment_intent = stripe.PaymentIntent.create(
  amount=1000,  # Amount in cents
  currency='usd',
)

# Return the client secret for the frontend
client_secret = payment_intent.client_secret




<script src="https://js.stripe.com/v3/"></script>

<button id="payButton">Pay</button>

<script>
    var stripe = Stripe('your_publishable_key');
    var payButton = document.getElementById('payButton');

    payButton.addEventListener('click', function () {
        fetch('/create-payment-intent', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            var clientSecret = data.client_secret;

            stripe.confirmCardPayment(clientSecret, {
                payment_method: {
                    card: cardElement,
                    billing_details: { name: 'John Doe' }
                }
            }).then(function(result) {
                if (result.error) {
                    console.log(result.error.message);
                } else {
                    alert('Payment succeeded!');
                }
            });
        });
    });
</script>




from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(_name_)
app.secret_key = 'your_secret_key'
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def _init_(self, id):
        self.id = id

users = {'user1': {'password': 'password123'}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return f'Hello {current_user.id}, welcome to your dashboard!'

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if _name_ == '_main_':
    app.run(debug=True)


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login</title>
</head>
<body>
    <form method="POST">
        <label for="username">Username</label>
        <input type="text" id="username" name="username">
        <br>
        <label for="password">Password</label>
        <input type="password" id="password" name="password">
        <br>
        <button type="submit">Login</button>
    </form>
</body>
</html>



import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return f'File uploaded: {filename}'

if _name_ == '_main_':
    app.run(debug=True)



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload Image</title>
</head>
<body>
    <h1>Upload an Image</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Upload</button>
    </form>
</body>
</html>



import pandas as pd

# Sample data
data = {'Name': ['John', 'Jane', 'Sam'], 'Age': [28, 34, 22]}
df = pd.DataFrame(data)

# Export to CSV
df.to_csv('data.csv', index=False)



import pandas as pd

# Sample data
data = {'Name': ['John', 'Jane', 'Sam'], 'Age': [28, 34, 22]}
df = pd.DataFrame(data)

# Export to Excel
df.to_excel('data.xlsx', index=False)


from flask import Flask, send_from_directory

app = Flask(_name_)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('uploads', filename)

if _name_ == '_main_':
    app.run(debug=True)
