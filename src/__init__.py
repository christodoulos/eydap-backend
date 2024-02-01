from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_swagger_ui import get_swaggerui_blueprint
from mongoengine import connect
from src.config import *

from src.blueprints.analysis import analysis
from src.blueprints.auth import auth

app = Flask(__name__)
jwt = JWTManager(app)

app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY

connect(alias=MONGO_DBNAME, db=MONGO_DBNAME, host=MONGO_HOST, port=MONGO_PORT)

cors = CORS(
    app,
    resources={
        r"*": {"origins": ["http://localhost:4200", "https://eydap-frontend.ddns.net"]}
    },
)

SWAGGER_URL = "/docs"
API_URL = "/static/swagger.json"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "Eydap tool",
    },
)

app.register_blueprint(analysis, url_prefix="/analysis")
app.register_blueprint(auth, url_prefix="/auth")
app.register_blueprint(swaggerui_blueprint)
