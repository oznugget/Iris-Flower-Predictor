#New additions to project by Shumirai Gunzo

import joblib
from data import ScMatrix
from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv

def main():

    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    model = joblib.load('iris_model.pkl')

    #request user input to predict species of flower
    #sepal- leaf-like strucutre forming the outer bottom part of a flower
    print("IRIS FLOWER SPECIES PREDICTOR")
    sepal_length = float(input("Please enter sepal length in cm"))
    sepal_width = float(input("Please enter sepal width in cm"))
    petal_length = float(input("Please enter petal length in cm"))
    petal_width = float(input("Please enter petal width in cm"))

    
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print("The predicted species of the flower is: ", prediction[0])

    #sepal graphs
    plt.scatter(dataset['sepal-length'], dataset['sepal-width'],color='gray', label='Dataset')
    plt.scatter(sepal_length, sepal_width,  color='red', s = 150, label='Predicted', marker='X')
    plt.title('Predicted Sepal Length and Width')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.show()

    #petal graphs
    plt.scatter(dataset['petal-length'], dataset['petal-width'],color='gray', label='Dataset')
    plt.scatter(petal_length, petal_width,  color='red', s = 150, label='Predicted', marker='X')
    plt.title('Predicted Petal Length and Width')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.legend()
    plt.show()  

    #data.py function to plot scatter matrix of the dataset
    ScMatrix()


if __name__ == "__main__":
    main()