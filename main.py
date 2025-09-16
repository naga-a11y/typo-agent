import os
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello from Cloud Run! Server is working.')

    def log_message(self, format, *args):
        print(format % args)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    server = HTTPServer(('', port), SimpleHandler)
    server.serve_forever()
