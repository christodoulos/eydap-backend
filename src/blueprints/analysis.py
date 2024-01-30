from flask import Blueprint, request, Response
from src.simulation import run
from flask_jwt_extended import get_jwt_identity, jwt_required, get_jwt
import json
import multiprocessing as mp


analysis = Blueprint("analysis", __name__)


@analysis.route("/execute", methods=["POST"])
@jwt_required()
def execute():
	# Get the user id from the JWT
	current_user = get_jwt_identity()
	claims = get_jwt()

	if not claims["isEnabled"]: return Response(response=json.dumps({"error": "User is not authorized to access this page"}), status=401)

	print(f"Current user: {current_user}, claims: {claims}")

	# Get the data from the request
	data = request.get_json()
	user_id, num1, num2 = data["user_id"], data["num1"], data["num2"]

	# Create a new process to run the simulation
	p = mp.Process(target=run, args=(user_id, num1, num2))
	p.start()
	print(f"Started process {p.pid}")

	# Return a response to the client
	return Response(
		response=json.dumps({"message": "Analysis started"}),
		status=200,
		mimetype="application/json",
	)