import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
import os
import sys


from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_tranier(self,train_array,test_array):
        try:
            logging.info('Spliting training and test input data')
            

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]


            # Create the dictinory of the models
            # Create the dictionary of models
            models = {
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # To get the best models score from the dict
            best_model_score=max(sorted(model_report.values()))

            # TO get best model name from dict

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            
            logging.info('Best found model on both training and testing dataset')


            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        
                        obj=best_model
                        )
            
            predicted=best_model.predict(X_test)

            r2_score1=r2_score(y_test,predicted)

            return r2_score1
        except Exception as e:
            raise CustomException(e,sys)