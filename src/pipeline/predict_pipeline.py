import sys
import pandas as pd

from src.utils import load_obj
from src.exception import Custon_exception

class Predict_pipeline:
    def __init__(self):
        pass

    def predict(self, feature):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\proprocessor.pkl'
            model = load_obj(file_path = model_path)
            preprocessor = load_obj(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(feature)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise Custon_exception(e, sys)

class Custon_data:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: int,
        reading_score: int,
        writing_score: int):
        
        self.race_ethnicity = race_ethnicity
        self.gender = gender
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custon_data_input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score': [self.writing_score]
            }
            return pd.DataFrame(custon_data_input_dict)
        except Exception as e:
            raise Custon_exception(e, sys)