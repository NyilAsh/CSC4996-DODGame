import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from AI.src.test import test_individual_input

def convert_numpy_to_python(obj):
    """Convert numpy types to regular Python types for JSON serialization"""
    if hasattr(obj, 'tolist'):  # For numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For numpy scalars
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj

class MyHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # Respond to preflight requests with CORS headers.
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == '/log_data':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                print("Received data from client:", data)
                
                # Get predictions from AI model
                predictions = test_individual_input(*data)
                print("AI model predictions (raw):", predictions)
                
                # Convert numpy types to Python types
                converted_predictions = convert_numpy_to_python(predictions)
                print("AI model predictions (converted):", converted_predictions)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                # Send predictions back to frontend
                response = {
                    'status': 'success',
                    'predictions': converted_predictions
                }
                print("Sending response to client:", response)
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                print("Error processing request:", str(e))
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                response = {'status': 'error', 'message': str(e)}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
    # Override to suppress logging of every request.
        return




def run(server_class=HTTPServer, handler_class=MyHandler, port=5001):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Extra processing server running on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
