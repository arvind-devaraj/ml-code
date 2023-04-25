from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

classifier_kNN = KNeighborsClassifier(n_neighbors = 3)

classifier_kNN.fit(X_traindata, Y_traindata)

Y_prediction = classifier_kNN.predict(X_testdata)

print("Accuracy of the model is :", metrics.accuracy_score(Y_testdata, Y_prediction))

#We provide sample data for the prediction of KNN 

Dummy_data = [[5, 5, 3,2], [4, 3, 6,3]]

pred_value = classifier_kNN.predict(Dummy_data)

prediction_species =[my_iris.target_names[q] for q in pred_value] 

print("Our model prediction is :", prediction_species) #outputs [versicolor, virginicia]

#https://thedatascientist.com/scikit-learn-101-exploring-important-functions/
