from flask import Blueprint, request, Response
from google.oauth2 import id_token
from google.auth.transport import requests
from flask_jwt_extended import create_access_token
from src.config import GOOGLE_AUDIENCE
from src.models.user import User
import json


auth = Blueprint("auth", __name__)


@auth.route("/google-auth", methods=["POST"])
def google_auth():
    idToken = request.json["idToken"]
    try: id_info = id_token.verify_oauth2_token(idToken, requests.Request(), GOOGLE_AUDIENCE)
    except ValueError: return Response({'error': 'Invalid user'}, status=401)

    user = User.get_user_by_google_id(id_info["sub"])

    if not user:
        user = User(
            email=id_info["email"],
            firstName=id_info["given_name"],
            lastName=id_info["family_name"],
            name=id_info["name"],
            googleId=id_info["sub"],
            photoUrl=id_info["picture"],
        )
        user.save()
    else:
        user.update(
            email=id_info["email"],
            firstName=id_info["given_name"],
            lastName=id_info["family_name"],
            name=id_info["name"],
            photoUrl=id_info["picture"],
        )
    
    additional_claims = {"isAdmin": user.isAdmin, "isEnabled": user.isEnabled}
    access_token = create_access_token(identity=user.email, additional_claims=additional_claims)

    return Response(json.dumps({"accessToken": access_token}), status=200)