
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os, io
import numpy as np
import pandas as pd
from Mediwatch_project import logger
from Mediwatch_project.pipeline.prediction import PredictionPipeline
from Mediwatch_project.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from Mediwatch_project.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from Mediwatch_project.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from Mediwatch_project.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
#from Mediwatch_project.pipeline.data_ingestion import DataIngestionTrainingPipeline
#from main import ModelTrainerTrainingPipeline, DataTransformationTrainingPipeline, DataValidationTrainingPipeline, DataIngestionTrainingPipeline   

app = Flask(__name__) # initializing a flask application

@app.route('/', methods=['GET']) # route to display the home page
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET']) # route to train the model
def train():
    os.system('python Mediwatch_project\pipeline\training.py')
    return "Training Successful!!"
    #return render_template('index.html', train_status = 'Model Trained Successfully!!')

@app.route('/predict', methods=['GET', 'POST']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        # try:
        #     #  reading the inputs given by the user
        #     age = int(request.form['age']) 
        #     gender = str(request.form['gender'])
        #     weight = float(request.form['weight'])

        #     print(f"Received inputs - Age, Gender, Weight: {age}, {gender}, {weight}")
        #     return render_template('index1.html', prediction_text = 'Inputs received successfully!')
        # finally:
        #     # creating a DataFrame from the inputs
        #    print("Successfully processed inputs data")

        if 'file' not in request.files:
            return jsonify(error="No file part"), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify(error="No selected file"), 400
        filename = secure_filename(f.filename)
        if not filename.lower().endswith('.csv'):
            return jsonify(error="Only .csv files allowed"), 400
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        data_validation_result = data_validation()
        if data_validation_result != "Data validation completed successfully!":
            return jsonify(error=data_validation_result), 400 
        data_transformation_result = data_transformation()
        if data_transformation_result != "Data transformation completed successfully!":
            return jsonify(error=data_transformation_result), 400
        DATA_TRANSFORMATION_PATH = os.path.join(app.config['TRANSFORMATION_FOLDER'], "train.csv")
        # model_trainer_result = model_trainer()
        # if model_trainer_result != "Model training completed successfully!":
        #     return jsonify(error=model_trainer_result), 400 
        # model_evaluation_result = model_evaluation()
        # if model_evaluation_result != "Model evaluation completed successfully!":
        #     return jsonify(error=model_evaluation_result), 400
        prediction_pipeline = PredictionPipeline()

        data = np.array(pd.read_csv(DATA_TRANSFORMATION_PATH))
        # Get 10 random unique row indices
        RESULES_SIZE = 10
        indices = np.random.choice(data.shape[0], size=RESULES_SIZE, replace=False)
        names = data[indices][:, 0] # First column as names
        print(f"Sampled Names: {str(names)}")

        sampled_rows = data[indices][:, 1:]

        #sampled_rows = data[:, 1:]

        predict = prediction_pipeline.predict(sampled_rows)

        print(f"Predictions: {str(predict)}")   

        return render_template('results.html', prediction = str(names) + str(predict))

        #return jsonify(prediction_pipeline.predict(sampled_rows)), 200

        # Send as file-like object
        # file_path = os.path.join(app.root_path, "artifacts/model_evaluation", "metrics.json")
        # return send_file(file_path, as_attachment=True)  # True = force download

        #return jsonify(success=True, saved_as=filename)
    elif request.method == 'GET':
        return render_template('file-upload.html', prediction_text = 'Please enter the inputs to get predictions.')
    
def data_validation():
    # This function can be used to validate the data
    # For now, it just returns a success message
    STAGE_NAME = "Data Validation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_validation = DataValidationTrainingPipeline()
        data_validation.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    return "Data validation completed successfully!"

def data_transformation():
    STAGE_NAME = "Data Transformation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_transformation = DataTransformationTrainingPipeline()
        data_transformation.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e
    return "Data transformation completed successfully!"

def model_trainer():
    STAGE_NAME = "Model Trainer stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        model_trainer = ModelTrainerTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e
    return "Model training completed successfully!"

def model_evaluation():
    STAGE_NAME = "Model evaluation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        model_evaluation = ModelEvaluationTrainingPipeline()
        model_evaluation.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e
    return "Model evaluation completed successfully!"


if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = 'artifacts/data_ingestion'
    app.config['TRANSFORMATION_FOLDER'] = 'artifacts/data_transformation'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True) # running the flask app in debug mode
    print("Flask app is running...")  # Debug message to indicate the app is running
        
                
            
            