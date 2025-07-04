import os 
import sys
from src.exception import Custon_exception
from src.logger import logging
from src.components.data_transformation import Data_transformation
from src.components.data_transformation import Data_tranformation_config
from src.components.model_training import Model_train_config
from src.components.model_training import Model_train

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class Data_ingestion_config:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class Data_ingestion:
    def __init__(self):
        self.ingestion_config = Data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or componenet")
        try:
            df = pd.read_csv('notebook/data/student.csv')
            logging.info("Read the dataset from dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header = True)

            train_set, test_set = train_test_split(df, test_size= 0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header=True)

            logging.info('Ingestion of the data completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
    
        except Exception as e:
            raise Custon_exception(e, sys)
        
if __name__ == "__main__":
    obj = Data_ingestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_trans = Data_transformation()
    train_arr, test_arr, preprocessor_path =  data_trans.inittiate_data_transformation(train_data, test_data)
    model_train = Model_train()
    print(model_train.initiate_model_train(train_arr, test_arr))