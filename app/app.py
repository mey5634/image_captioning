from flask import Flask, request, jsonify, json
from model_server import ModelServer
from predict import predict

def create_app():
    # create and configure the app
    app = Flask(__name__)
    app.config.from_mapping(
        MODEL=ModelServer()
    )

    # our single endpoint: receives {'image': 'abs/path/to/image'}, returns {'caption': 'some caption'}
    @app.route('/', methods = ['POST'])
    def api_root():
        if request.headers['Content-Type'] == 'application/json':
            img_path = request.json['image']
            message = {
                'status': 200,
                'caption': predict(app.config['MODEL'], img_path)
            }
            resp = jsonify(message)
            resp.status_code = 200
        else:
            message = {
                    'status': 415,
                    'caption': 'Unsupported media type'
            }
            resp = jsonify(message)
            resp.status_code = 415
        return resp
    return app

if __name__ == '__main__':
    create_app().run()
