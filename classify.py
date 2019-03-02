import pandas
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


if __name__=="__main__":
    filename = "irisdata.csv"
    headers = ['sepal_length','sepal_width','petal_length','petal_width','class']
    dataset = pandas.read_csv(filename,names=headers)
    
    # to get the dimensions of the data
    print(dataset.shape)

    # to get the first 10 rows
    # print(dataset.head(10))

    # to get statistical info
    # print(dataset.describe()) 

    # to get class dist
    # print(dataset.groupby('class').size())

    # create plots
    # dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    # plt.show()

    # histograms
    # dataset.hist()
    # plt.show()

    # scatter_matrix(dataset)
    # plt.show()

    array = dataset.values
    dimensions = array[:,0:4]
    class_names = array[:,4]
    
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(dimensions, class_names, test_size=validation_size, random_state=seed)
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        print("-"*30)

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))