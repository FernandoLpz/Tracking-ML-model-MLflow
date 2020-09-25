import os
import sys
import mlflow
import logging
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass

@dataclass
class Parameters:
   # Preprocessing parameeters
   data_name: str
   data_path: str = f"data"
   k_folds: int = 10
   test_size: float = 0.1

class Pipeline:
   def __init__(self, params):
      self.params = params
      self.dataset_path = f"{params.data_path}/{params.data_name}.csv"
      self.dataset = None
      self.clf = None
      self.x = None
      self.y = None
      self.x_train = None
      self.x_test = None
      self.y_train = None
      self.y_test = None
      
      self.best_max_depth = None
      self.best_grid = None
      self.best_splitter = None
      self.tree = None

   def load_data(self):
      # Logging
      logging.info(f"Loading file at {self.dataset_path}")
      
      # Read dataset
      self.dataset = pd.read_csv(self.dataset_path)
      
      # Save artifact
      mlflow.log_artifact(f"{self.dataset_path}")
      
   def preprocessing(self):
      # Logging
      logging.info(f"Preprocessing dataset")
   
      # Encoding categorical features
      columns_to_be_encoded = self.dataset.drop(['Class'], axis=1).columns
      self.x = pd.get_dummies(self.dataset.drop(['Class'], axis=1), columns=columns_to_be_encoded)
      
      # Encoding target
      classes = self.dataset['Class'].unique()
      for idx, class_name in enumerate(classes):
         self.dataset['Class'] = self.dataset['Class'].replace(class_name, idx)
      self.y = self.dataset['Class']

   def split_data(self):
      # Logging
      logging.info(f"Split data into train and test")
      self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, test_size=self.params.test_size)
      
   def parameter_tuning(self):
      # Logging
      logging.info(f"Applying grid search for parameter tuning")
      
      # Defining parameters grid
      parameters = {'criterion': ['gini','entropy'], 'splitter': ['best','random'], 'max_depth': [2,3,4]}
      
      # Grid search
      tree = DecisionTreeClassifier()
      grid = GridSearchCV(tree, parameters, cv=self.params.k_folds)
      grid.fit(self.x_train, self.y_train)
      
      self.best_max_depth = grid.best_params_['max_depth']
      self.best_criterion = grid.best_params_['criterion']
      self.best_splitter = grid.best_params_['splitter']
         
      mlflow.log_param(f'best_max_depth', self.best_max_depth)
      mlflow.log_param(f'best_criterion', self.best_criterion)
      mlflow.log_param(f'best_splitter', self.best_splitter)
      
   def k_fold_cross_validation(self):
      # Logging
      logging.info(f"Applying k-fold cross validation")
      
      self.tree = DecisionTreeClassifier(max_depth=self.best_max_depth, splitter=self.best_splitter, criterion=self.best_criterion)
      kfold_scores = cross_val_score(self.tree, self.x_train, self.y_train, cv=self.params.k_folds)
      
      mlflow.log_metric(f"average_accuracy", kfold_scores.mean())
      mlflow.log_metric(f"std_accuracy", kfold_scores.std())
      
   def model_evaluation(self):
      logging.info(f"Model evaluation")
      
      self.tree.fit(self.x_train, self.y_train)

      mlflow.log_metric(f"train_accuracy", self.tree.score(self.x_train, self.y_train))
      mlflow.log_metric(f"test_accuracy", self.tree.score(self.x_test, self.y_test))
      
   def save_model(self):
      logging.info(f"Saving model")
      mlflow.sklearn.save_model(self.tree, f"model/mymodel_3", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)


if __name__ == '__main__':
   # Init logging
   logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO, filename='logs/example.log')
   
   # If data name is provided by commanda line, success
   # If data name is not provided, shows an exception
   try:
      # Gets data name from command line
      data_name = sys.argv[1]
   except:
      print(f"You must provide a dataname, please try:\npython main.py [dataname]")

   # Init parameters
   params = Parameters(data_name)
   if os.path.isfile(f"{params.data_path}/{params.data_name}.csv"):

      # Initi Mlflow client
      client = MlflowClient()
      
      # If the project does not exists, it creates a new one
      # If the project already exists, it is taken the project id
      try:
         # Creates a new experiment
         experiment_id = client.create_experiment(data_name)
         logging.info(f"The experiment {data_name} was created with id={experiment_id} ")
      except:
         # Retrieves the experiment id from the already created project
         experiment_id = client.get_experiment_by_name(data_name).experiment_id
         logging.info(f"The id={experiment_id} from experiment {data_name} was retrieved successfully")
      
      # Initialize mlflow context
      with mlflow.start_run(experiment_id=experiment_id):
         # Pipeline execution
         pipeline = Pipeline(params)
         pipeline.load_data()
         pipeline.preprocessing()
         pipeline.split_data()
         pipeline.parameter_tuning()
         pipeline.k_fold_cross_validation()
         pipeline.model_evaluation()
         pipeline.save_model()