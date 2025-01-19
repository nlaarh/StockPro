from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World! If you can see this, the web server is working!'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
