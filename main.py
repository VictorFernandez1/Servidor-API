from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
import pandas as pd
import io
import os
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from importlib import import_module

@asynccontextmanager                # Lifespan manager to handle startup and shutdown events. Global variables for model, encoder and scaler.
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.label_encoder = None
    app.state.scaler = None
    app.state.model_folder = None
    yield

app = FastAPI(lifespan=lifespan)


# The next endpoint is used for sending the list of tensorflow models. The client will ask for a list of available models. Available models are in folder "models".
@app.get("/list-models/")                           # Endpoint to list available models
async def list_models():
    # List all files and folders in the "models" directory, except gitignore file

    models = [f for f in os.listdir("Tf_Models") if f != ".gitignore"]
    return {"models": models}


# The next endpoint is used for selecting a model. The client will send the name of the model to be used.
@app.post("/select-model/")                           # Endpoint to select a model
async def select_model(model_name: str = Form(...)):
    # Check if the model exists
    if not os.path.exists(os.path.join("Tf_Models", model_name)):
        return {"message": "Model not found"}

    # Load the tensorflow model. The model is a keras file that is inside a folder with the name of the model. The name of the model starts with Classifier.

    model_folder = os.path.join("Tf_Models", model_name)
    #Now we will load the model, the encotder and the scaler.
    try:
        model_path = os.path.join(model_folder, [f for f in os.listdir(model_folder) if f.startswith("Classifier")][0])
        encoder_path = os.path.join(model_folder, [f for f in os.listdir(model_folder) if f.startswith("Encoder")][0])
        scaler_path = os.path.join(model_folder, [f for f in os.listdir(model_folder) if f.startswith("Scaler")][0])
    except IndexError:
        return {"message": "Model files not found in the specified folder"}

    app.state.model_folder = model_folder
    app.state.model = tf.keras.models.load_model(model_path)
    app.state.label_encoder = joblib.load(encoder_path)
    app.state.scaler = joblib.load(scaler_path)

    return {"message": f"Model {model_name} selected"}

@app.post("/upload-csv/")                           # Endpoint to upload CSV files and make predictions
async def upload_csv(file: UploadFile = File(...)):
    # Check if it's a CSV
    if not file.filename.endswith(".csv"):
        return {"message": "Only CSV files are allowed"}

    # Let´s convert the csv_file to a pandas dataframe

    df_experiment = pd.read_csv(io.StringIO((await file.read()).decode('utf-8')))

    # Now we will preprocess the data using the fuction Preprocesa from the python file we can find in the model folder. The name of the file Preprocesa_(folder_name).py. The name of the function is Preprocesa_(folder_name).


    module_name = f"Tf_Models.{os.path.basename(app.state.model_folder)}.Preprocesa_{os.path.basename(app.state.model_folder)}"
    try:
        module = import_module(module_name)  # Import the module dynamically
        preprocesa_function = getattr(module, f"Preprocesa_{os.path.basename(app.state.model_folder)}")  # Get the function dynamically
    except (ModuleNotFoundError, AttributeError) as e:
        return {"message": f"Preprocessing function not found: {e}"}

    X = preprocesa_function(df_experiment)  # Call the function with df_ex


    # if X is empty or none, return a message
    if X is None or X.empty:
        return {"message": "No valid data to predict"}

    # Hacemos la predicción, puede que se devuelvan varios resultados

    predictions = app.state.model.predict(app.state.scaler.transform(X))
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    predicted_labels = app.state.label_encoder.inverse_transform(predicted_classes)
    # Return 2 lists: predictions and predicted labels

    return {"predictions": predictions.tolist(), "predicted_labels": predicted_labels.tolist()}


@app.get("/ping")                         # Lightweight endpoint for uptime checks
async def ping():
    """
    Lightweight endpoint for uptime checks.
    Returns 200 OK quickly to keep the server awake.
    """
    return {"status": "ok", "message": "Server is alive"}




