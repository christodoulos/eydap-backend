from flask import Flask
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from mongoengine import connect
from src.config import *

from src.blueprints.analysis import analysis


SWAGGER_URL = "/docs"
API_URL = "/static/swagger.json"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "Digital Twin of Attica API",
    },
)

app = Flask(__name__)

connect(alias=MONGO_DBNAME, db=MONGO_DBNAME, host=MONGO_HOST, port=MONGO_PORT)

cors = CORS(
    app,
    resources={
        r"*": {"origins": ["http://localhost:4200"]}
    },
)

app.register_blueprint(analysis, url_prefix="/analysis")

app.register_blueprint(swaggerui_blueprint)
