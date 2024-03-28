from flask import Flask
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

def predict_iris(features):
    predicted_iris_type = clf.predict([features])[0]
    return 'setosa' if predicted_iris_type == 0 else ('versicolor' if predicted_iris_type == 1 else 'virginica')
