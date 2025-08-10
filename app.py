from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from Mediwatch_project.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask application