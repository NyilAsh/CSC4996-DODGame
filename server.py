import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from AI.src.test import test_individual_input

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
                # Print only the arrays received via POST.
                pred=test_individual_input(*data)
                print(data,'\n')
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                response = {'status': 'success'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
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
