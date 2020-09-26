import mlflow.sklearn

sk_model = mlflow.sklearn.load_model("model/sklearn_mushroom")
print(sk_model)
