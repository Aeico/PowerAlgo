from flask import Flask, request, jsonify


choice = {"choice" : -1}

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/')
    def index():
        return "Main Page"

    @app.route('/choice',methods = ['POST', 'GET'])
    def choice():
        global choice

        if request.method == 'POST':
            resp = request.json
            choice = resp
            return jsonify(choice) 
        if request.method == 'GET':
            return jsonify(choice)
    
    return app

if __name__ == "__main__":
    create_app().run
