from flask import Flask

def classifier(input):
    return "label"

app = Flask(__name__)
@app.route('/test/<name>')
def hello_world(name='No name'):
    return 'Hello, '+name

@app.route('/classifier/')
def call_classifier():
    foto = None # request get picture
    out = classifier(foto)
    return out