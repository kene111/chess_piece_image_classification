import os
from flask import Flask
from flask_cors import CORS
from .config.deploy_config import config
from .utils.sys_utils import load_saved_model
from .config.sys_config import  ImageClassConfig

def create_app(env_config=None):
    # instantiate the app
    app = Flask(__name__)
    cors = CORS(app)

    ###### ENVIROMENT VARIABLE CONFIGURATION #######################
    if env_config is None:
        env_config = os.getenv("PROD_APP_SETTINGS", "development")
    app.config.from_object(config[env_config])
    ###### MODEL INITIALIZATION #####################################
    ImageClassConfig.classification_model = load_saved_model(ImageClassConfig.model_path)
    ##### IC BLUEPRINT ENDPOINTS ################
    from .ic_connect import image_class_endpoints
    app.register_blueprint(image_class_endpoints)

    def ctx():
        return {"app": app}
    return app
