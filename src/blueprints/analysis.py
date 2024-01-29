from flask import Blueprint, request, Response
import json
import multiprocessing as mp
from src.simulation import run


analysis = Blueprint("analysis", __name__)


@analysis.route("/execute", methods=["POST"])
def execute():
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