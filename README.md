# Chess Piece Image Classification Task

Image Classification Project.

### Repository Breakdown:

##### INDSIDE ```ROOT``` FOLDER
1. ```image_class_client```: Contains the steamlit code.
2. ```image_class_server```: Contains the flask server.
3. ```notebooks```: Contains training code for classification model.
4. ```Docker-compose.yaml```: Docker compose file



##### INSIDE THE ```image_class_client``` FOLDER :
1. ```src/components```: This folder contains the key features used by image classification system.
     1. ```classification.py```: This module classifies images
2. ```app.py```: This module runs the flask app.
3. ```src/utils```: Contains utility scripts used through out the ML system.
      1. ```sys_utils.py```:This module holds general utility functions.
4. ```ic_connect.py```: This module integrates flasks blueprint.
5. ```src/config```: This folder contains the necessary config files.
      1. ```deploy_config.py```: This module holds deployment configuration.
      2. ```system_config.py```: This module holds system configuration.
6. ```Dockefile```: Dockerfile
7. ```start_server```: Bash file for starting system via docker compose.


##### INSIDE THE ```image_class_client``` FOLDER :
1. ```start_client```: Bash file for starting system via docker compose
2. ```user_interface.py```: This module creates an interface with streamlit.
3. ```requirements.txt```: This contains the requirements configuration.
4.  ```Dockefile```: Dockerfile

##### INSIDE THE ```notebook``` FOLDER :
1. ```Image_classification.ipynb```: Image classification training notebook


### How to run the System locally:
### Without Docker Compose
1. Create and activate a two different virtual environment for the client and server system on different terminals: ```python3 -m venv _name_of_virtual_env_```.
2. Install dependencies using: ```pip install -r requirements.txt```.
  
3. Run the server locally using: ```python app.py```.
4. Run the client locally using: ```streamlit run user_interface.py```.
5. Once Successful a screen should appear with an easy to user UI.

### With Docker Compose:
1. Change directory to root folder.
2. run "docker-compose build".
3. then f "docker-compose up".


### API ENDPOINTS
1. classification_endpoint (```POST```) : ```http://127.0.0.1:5000/classify```;```TESTED```.
2. system_alive_endpoint (```GET```) : ```http://127.0.0.1:5000/ic_alive```;```TESTED```.



## API REQUESTS BODY EXAMPLES FOR EACH ENDPOINTS
1. classification_endpoint (```POST```) :
```
{
  'img_b64': b64_string
}
```


### API RESPONSE STRUCTURE BREAKDOWN:
#### classification_endpoint (```POST```) :
```
{
    "system_response": {
                          "target":target,
                          "confidence": confidence_score
                        }
}
```

#### system_alive_endpoint (```GET```) :
```
{
   "system_response":"System is running!"
}
```
