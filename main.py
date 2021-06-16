import pandas as pnd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import KFold
import itertools
import numpy as nmp
import seaborn as sbn
import pickle

# reading data files and storing it into dataf.
dataf = pnd.read_csv("fake_or_real_news.csv")

#Shape function returns rows & column of data file(csv file) in an array.
dataf.shape

# head is used to print first 5 lines. 
dataf.head()


# by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes.
def cdn(dataFile):
    return sbn.countplot(x='label', data=dataFile, palette='hls')

cdn(dataf)


#checking null values in data set(csv file).
def dqualityCheck():
    print("Checking data qualitites...")
    dataf.isnull().sum()
    dataf.info()  
    print("check finished.")
dqualityCheck()



#we need to seperate x & y.

#In y we had set label column of csv file.
y = dataf.label
y.head()

#In x we had set text column of csv file.
x = dataf.text
x.head()

# Drop the 'label' column.
dataf.drop("label", axis=1)


# training and testing the data sets.
X_train, X_test, y_train, y_test = train_test_split(dataf['text'], y, test_size=0.33, random_state=53)
X_train.head()
X_test.head()


# we will start with simple bag of words technique 
# Building the Count and Tfidf Vectors

# Initialize the 'count_vectorizer' & checkng StopWords some example of stopwords are the, is ,am , are etc.
cv = CountVectorizer(stop_words='english')
count_train = cv.fit_transform(X_train)
print(cv)
print(count_train)



# print training doc term matrix
# we have matrix of size of (4244, 56922) by calling below

def get_countVectorizer_stats():
    
    #vocab size
    print(count_train.shape)

    #check vocabulary using below command
    print(cv.vocabulary_)

get_countVectorizer_stats()


# Transform the test set
count_test = cv.transform(X_test)



# create tf-idf frequency features
# This will removes words which appear more than 70% of the articles

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)





# Fit and transform train set, transform test set
# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
def get_tfidf_stats():
    tfidf_train.shape
    #get train data feature names 
    print(tfidf_train.A[:10])

get_tfidf_stats()




# Transform the test set 
tfidf_test = tfidf_vectorizer.transform(X_test)

# get feature names
# Get the feature names of 'tfidf_vectorizer'
print(tfidf_vectorizer.get_feature_names()[-10:])

# Get the feature names of 'cv'
print(cv.get_feature_names()[:10])
count_df = pnd.DataFrame(count_train.A, columns=cv.get_feature_names())
tfidf_df = pnd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)


# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))
print(count_df.head())
print(tfidf_df.head())




# Function to plot the confusion matrix
# Model can never be 100% accurate, it basically depends on 4 things
#True Positive: When News is true and it shows true.
#True Negative: When news is true but it shows fake.
#False Positive: When news is fake but it shows true.
#False Negative: When news is fake but it shows fake.


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = nmp.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, nmp.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




# building classifier using naive bayes 
# Naive Bayes classifier for Multinomial model

nb_pipeline = Pipeline([
        ('NBTV',tfidf_vectorizer),
        ('nb_clf',MultinomialNB())])




# Fit Naive Bayes classifier according to X, y
nb_pipeline.fit(X_train,y_train)



# Perform classification on an array of test vectors X
predicted_nbt = nb_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbt)
print(f'Accuracy: {round(score*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_nbt, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)
nbc_pipeline = Pipeline([
        ('NBCV',cv),
        ('nb_clf',MultinomialNB())])
nbc_pipeline.fit(X_train,y_train)
predicted_nbc = nbc_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbc)
print(f'Accuracy: {round(score*100,2)}%')
cm1 = metrics.confusion_matrix(y_test, predicted_nbc, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm1, classes=['FAKE', 'REAL'])
print(cm1)
print(metrics.classification_report(y_test, predicted_nbt))
print(metrics.classification_report(y_test, predicted_nbc))



# building Passive Aggressive Classifier 
linear_clf = Pipeline([
        ('linear',tfidf_vectorizer),
        ('pa_clf',PassiveAggressiveClassifier(max_iter=50))])
linear_clf.fit(X_train,y_train)


#Predict on the test set and calculate accuracy
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print(f'Accuracy: {round(score*100,2)}%')



#Build confusion matrix
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)
print(metrics.classification_report(y_test, pred))


# saving best model to the disk
model_file = 'final_model1.pkl'
pickle.dump(linear_clf,open(model_file,'wb'))

import pickle
var = input("Please enter the news text you want to verify: ")
print("You have entered: " + str(var))


#function to run for prediction
def detecting_fake_news(var):    
#retrieving the best model for prediction call
    load_model = pickle.load(open('final_model1.pkl', 'rb'))
    prediction = load_model.predict([var])
    #prob = load_model.predict_proba([var])

    return (print("The given statement is ",prediction[0]))


if __name__ == '__main__':
    detecting_fake_news(var)
