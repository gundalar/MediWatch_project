
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

from markupsafe import escape
import csv
app = Flask(__name__) # initializing a flask application
dataTransformationTrainingPipeline = DataTransformationTrainingPipeline(None)

# @app.route('/', methods=['GET']) # route to display the home page
# def home():
#     return render_template('index.html')


@app.route('/train', methods=['GET']) # route to train the model
def train():
    #os.system('python Mediwatch_project\pipeline\training.py')
    model_trainer_result = model_trainer()
    if model_trainer_result != "Model training completed successfully!":
        return jsonify(error=model_trainer_result), 400 
    return "Training Successful!!"
    #return render_template('index.html', train_status = 'Model Trained Successfully!!')

@app.route('/predict', methods=['GET', 'POST']) # route to show the predictions in a web UI
def index_something():
    if request.method == 'POST':
        try:
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
            #dataTransformationTrainingPipeline.transformer.dropped_data
            print(f"DataTransformer attributes: {dir(dataTransformationTrainingPipeline.transformer)}")
            results = dataTransformationTrainingPipeline.transformer.attach_dropped_data(pd.DataFrame(data))
            print(f"Data Sample: {str(results.head(5))}")
            # Get 10 random unique row indices
            RESULES_SIZE = 100
            indices = np.random.choice(data.shape[0], size=RESULES_SIZE, replace=False)
            patient_nbrs = results['patient_nbr'][indices] # First column as names
            print(f"Patient Numbers: {str(patient_nbrs)}")

            sampled_rows = data[indices][:, 1:]

            #sampled_rows = data[:, 1:]

            predict = prediction_pipeline.predict(sampled_rows)

            print(f"Predictions: {str(predict)}")   

            predictions = [{"patient_nbr" : str(patient_nbr) , "result" : str(pred)} for patient_nbr, pred in zip(patient_nbrs, predict)]

            #predictions = [dict(zip(names, pred)) for pred in predict] #{"patient_id" + ":" + str(name) , "result" + ": " + str(pred)} for name, pred in zip(names, predict)]

            return render_template('results.html', predictions = predictions)

            #return jsonify(prediction_pipeline.predict(sampled_rows)), 200

            # Send as file-like object
            # file_path = os.path.join(app.root_path, "artifacts/model_evaluation", "metrics.json")
            # return send_file(file_path, as_attachment=True)  # True = force download

            #return jsonify(success=True, saved_as=filename)
        except Exception as e:
            print(f"Error occurred: {e}")
            return jsonify(error=str(e)), 500   
    elif request.method == 'GET':
        return render_template('file-upload.html', prediction_text = 'Please enter the inputs to get predictions.')


    def render_table(rows):
        if not rows:
            return "<em>No data</em>"
        parts = ['<div class="table-wrap">', '<table class="tbl">']
        for i, row in enumerate(rows):
            tag = 'th' if i == 0 else 'td'
            parts.append('<tr>')
        for cell in row:
            parts.append(f'<{tag}>' + escape(cell) + f'</{tag}>')
            parts.append('</tr>')
            parts.append('</table></div>')
        return ''.join(parts)


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     table_html = None
#     error = None
#     if request.method == 'POST': # non‑JS fallback; re-render same page
#         data = request.form.get('data', '')
#         try:
#             rows = list(csv.reader(io.StringIO(data)))
#             table_html = render_table(rows)
#         except Exception as e:
#             error = str(e)
#     return render_template('prediction-table.html', table_html=table_html, error=error)

# @app.post('/api/table')
# def api_table(): # JS path; returns only the HTML fragment for the table
#     data = request.form.get('data', '')
#     rows = list(csv.reader(io.StringIO(data)))
#     return jsonify({"html": render_table(rows)})

@app.route('/predictions', methods=['GET'])
def index():
    # Example dynamic data: a list of dictionaries
    predictions = [
        {"id": 1, "name": "Alice", "role": "Engineer", "age": 30},
        {"id": 2, "name": "Bob", "role": "Manager", "age": 40},
        {"id": 3, "name": "Charlie", "role": "Analyst", "age": 28},
    ]
    # Pass it to the template
    return render_template("results.html", predictions=predictions)
    
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
        # data_transformation = DataTransformationTrainingPipeline()
        # self.transformer = data_transformation
        dataTransformationTrainingPipeline.main()
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
        
                
            
            