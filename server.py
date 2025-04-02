from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ai_predict', methods=['POST'])
def ai_predict():
    data = request.get_json()
    print("Received AI prediction request with data:", data)
    return jsonify({"status": "success", "message": "AI prediction received", "data": data})

if __name__ == '__main__':
    app.run(debug=True)