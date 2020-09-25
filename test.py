import mlflow.sklearn

sk_model = mlflow.sklearn.load_model("model/mymodel")
print(sk_model)
