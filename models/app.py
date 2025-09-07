from flask import Flask, request, jsonify
from prediction_finale import predict_next_leave

app = Flask(__name__)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Get employee name from query parameter or JSON
    if request.method == "POST":
        data = request.json
        name = data.get("Nom_Employe", "")
    else:
        name = request.args.get("Nom_Employe", "")

    if not name:
        return jsonify({"error": "Nom_Employe manquant"}), 400

    result = predict_next_leave(name)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
