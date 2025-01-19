from flask import Flask, Response
import requests

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def proxy(path):
    # Forward the request to Streamlit
    resp = requests.get(f'http://localhost:8501/{path}')
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]
    
    return Response(resp.content, resp.status_code, headers)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
