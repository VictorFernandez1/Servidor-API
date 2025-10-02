from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
import pandas as pd
import io
import os
from codigos import preprocesa
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler

@asynccontextmanager                # Lifespan manager to handle startup and shutdown events. Global variables for model, encoder and scaler.
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.label_encoder = None
    app.state.scaler = None
    yield

app = FastAPI(lifespan=lifespan)


# The next endpoint is used for sending the list of tensorflow models. The client will ask for a list of available models. Available models are in folder "models".
@app.get("/list-models/")                           # Endpoint to list available models
async def list_models():

    models = [f for f in os.listdir("models")]
    return {"models": models}


# The next endpoint is used for selecting a model. The client will send the name of the model to be used.
@app.post("/select-model/")                           # Endpoint to select a model
async def select_model(model_name: str = Form(...)):
    # Check if the model exists
    if not os.path.exists(os.path.join("models", model_name)):
        return {"message": "Model not found"}

    # Load the tensorflow model. The model is a keras file that is inside a folder with the name of the model. The name of the model starts with Classifier.

    model_folder = os.path.join("models", model_name)
    #Now we will load the model, the encotder and the scaler.
    try:
        model_path = os.path.join(model_folder, [f for f in os.listdir(model_folder) if f.startswith("Classifier")][0])
        encoder_path = os.path.join(model_folder, [f for f in os.listdir(model_folder) if f.startswith("Encoder")][0])
        scaler_path = os.path.join(model_folder, [f for f in os.listdir(model_folder) if f.startswith("Scaler")][0])
    except IndexError:
        return {"message": "Model files not found in the specified folder"}


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

    X=preprocesa(df_experiment)

    # if X is empty or none, return a message
    if X is None or X.empty:
        return {"message": "No valid data to predict"}

    # Hacemos la predicción, puede que se devuelvan varios resultados

    predictions = app.state.model.predict(app.state.scaler.transform(X))
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    predicted_labels = app.state.label_encoder.inverse_transform(predicted_classes)
    # Return 2 lists: predictions and predicted labels

    return {"predictions": predictions.tolist(), "predicted_labels": predicted_labels.tolist()}




