import os
import logging
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Creatw Flask app
app= Flask(__name__)
app.secret_key = os.environ["SESSION_SECRET"]
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
app.config["UPLOAD_FOLDER"] = "/tmp/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # 16MB max file size

# Upload directory if it exists or not
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# Import routes
from routes import *

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=False, port=5000)