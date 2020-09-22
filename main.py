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

class Pipeline:
   def __init__(self, dataset_path):
      self.dataset_path = dataset_path
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
      logging.info(f"{self.dataset_path}")
      mlflow.log_artifact(f"{self.dataset_path}")
      
      # Read dataset
      self.dataset = pd.read_csv(self.dataset_path)
      
   def preprocessing(self):
      # Encoding categorical features
      columns_to_be_encoded = self.dataset.drop(['Class'], axis=1).columns
      self.x = pd.get_dummies(self.dataset.drop(['Class'], axis=1), columns=columns_to_be_encoded)
      
      # Encoding target
      classes = self.dataset['Class'].unique()
      for idx, class_name in enumerate(classes):
         self.dataset['Class'] = self.dataset['Class'].replace(class_name, idx)
      self.y = self.dataset['Class']

   def split_data(self):
      logging.info(f"splitting data")
      self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, test_size=0.2)
      
   def parameter_tuning(self):
      logging.info(f"training model")
      
      # Defining parameters grid
      parameters = {'criterion': ['gini','entropy'], 'splitter': ['best','random'], 'max_depth': [2,3,4]}
      
      # Grid search
      tree = DecisionTreeClassifier()
      grid = GridSearchCV(tree, parameters, cv=5)
      grid.fit(self.x_train, self.y_train)
      
      self.best_max_depth = grid.best_params_['max_depth']
      self.best_criterion = grid.best_params_['criterion']
      self.best_splitter = grid.best_params_['splitter']
         
      mlflow.log_param(f'best_max_depth', self.best_max_depth)
      mlflow.log_param(f'best_criterion', self.best_criterion)
      mlflow.log_param(f'best_splitter', self.best_splitter)
      mlflow.log_metric('best_score', grid.best_score_)
      
   def k_fold_cross_validation(self):
      self.tree = DecisionTreeClassifier(max_depth=self.best_max_depth, splitter=self.best_splitter, criterion=self.best_criterion)
      kfold_scores = cross_val_score(self.tree, self.x_train, self.y_train, cv=5)

      print(f"Average accuracy: {kfold_scores.mean()}")
      print(f"Std accuracy: +/-{kfold_scores.std()}")
   
      
   def model_evaluation(self):
      logging.info(f"calculating metrics")
      
      self.tree.fit(self.x_train, self.y_train)
      print(f"\nTrain accuracy: {self.tree.score(self.x_train, self.y_train)}")
      print(f"Test accuracy: {self.tree.score(self.x_test, self.y_test)}")


if __name__ == '__main__':
   dataset_path = sys.argv[1]
   
   # Init logging
   logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
   
   # Initi Mlflow client
   client = MlflowClient()
   idx = client.create_experiment("mushroom")
   # experiment = client.get_experiment_by_name("mushroom")
   
   with mlflow.start_run(experiment_id=idx):
      pipeline = Pipeline(dataset_path)
      pipeline.load_data()
      pipeline.preprocessing()
      pipeline.split_data()
      pipeline.parameter_tuning()
      pipeline.k_fold_cross_validation()
      pipeline.model_evaluation()