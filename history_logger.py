from flask_cors import CORS
from flask import Flask, request, jsonify
import csv
import os
 
app = Flask(__name__)
CORS(app) 

CSV_FILE = os.path.abspath('AI/src/data/data.csv')
print(f"Writing to: {CSV_FILE}")
# Create CSV with exact headers if doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = [
            # Attacker A history
            "attacker A's prev1 x", "attacker A's prev1 y",
            "attacker A's prev2 x", "attacker A's prev2 y",
            "attacker A's prev3 x", "attacker A's prev3 y",
            # Attacker B history
            "attacker B's prev1 x", "attacker B's prev1 y",
            "attacker B's prev2 x", "attacker B's prev2 y",
            "attacker B's prev3 x", "attacker B's prev3 y",
            # Attacker C history
            "attacker C's prev1 x", "attacker C's prev1 y",
            "attacker C's prev2 x", "attacker C's prev2 y",
            "attacker C's prev3 x", "attacker C's prev3 y",
            # Defender A history
            "defender A's prev1 x", "defender A's prev1 y",
            "defender A's prev2 x", "defender A's prev2 y",
            "defender A's prev3 x", "defender A's prev3 y",
            # Defender B history
            "defender B's prev1 x", "defender B's prev1 y",
            "defender B's prev2 x", "defender B's prev2 y",
            "defender B's prev3 x", "defender B's prev3 y",
            # Current positions
            "attacker A current x", "attacker A current y",
            "attacker B current x", "attacker B current y",
            "attacker C current x", "attacker C current y",
            "defender A current x", "defender A current y",
            "defender B current x", "defender B current y"
        ]
        writer.writerow(headers)

@app.route('/log_history', methods=['POST'])
def log_history():
    try:
        data = request.json
        
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)