import re, nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import joblib as joblib
from sklearn.naive_bayes import MultinomialNB

 

def normalizer(reviews): # Cleaning Reviews
    soup = BeautifulSoup(reviews, 'lxml')   # Removing HTML encoding such as ‘&amp’,’&quot’
    souped = soup.get_text()
    only_words = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\\\S+)"," ", souped) # removing @mentions, hashtags, urls.

 
 

    tokens = nltk.word_tokenize(only_words)
    removed_letters = [word for word in tokens if len(word)>2] # removing words with length less than or equal to 2
    lower_case = [l.lower() for l in removed_letters]

 

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

 

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

 

def Cross_validation(data, targets, tfidf, clf_cv, model_name): # Performs cross-validation on SVC

 
    kf = KFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
    scores=[]
    data_train_list = []
    targets_train_list = []
    data_test_list = []
    targets_test_list = []
    iteration = 0
    print("Performing cross-validation for {}...".format(model_name))
    for train_index, test_index in kf.split(data):
        iteration += 1
        print("Iteration ", iteration)
        data_train_cv, targets_train_cv = data[train_index], targets[train_index]
        data_test_cv, targets_test_cv = data[test_index], targets[test_index]
        data_train_list.append(data_train_cv) # appending training data for each iteration
        data_test_list.append(data_test_cv) # appending test data for each iteration
        targets_train_list.append(targets_train_cv) # appending training targets for each iteration
        targets_test_list.append(targets_test_cv) # appending test targets for each iteration
        tfidf.fit(data_train_cv) # learning vocabulary of training set
        data_train_tfidf_cv = tfidf.transform(data_train_cv)
        print("Shape of training data: ", data_train_tfidf_cv.shape)
        data_test_tfidf_cv = tfidf.transform(data_test_cv)
        print("Shape of test data: ", data_test_tfidf_cv.shape)
        clf_cv.fit(data_train_tfidf_cv, targets_train_cv) # Fitting SVC
        predicted_target_test = clf_cv.predict(data_test_tfidf_cv)
        tn, fp, fn, tp = confusion_matrix(targets_test_cv, predicted_target_test).ravel()
        precision_score = tp / (tp + fp)        
        scores.append(precision_score) # appending cross-validation precision score for each iteration
    print("List of cross-validation precision scores for {}: ".format(model_name), scores)
    mean_precision_score = np.mean(scores)
    print("Mean cross-validation precision for {}: ".format(model_name), mean_precision_score)
    print("Best cross-validation precision for {}: ".format(model_name), max(scores))
    max_pres_index = scores.index(max(scores)) # best cross-validation accuracy
    max_pres_data_train = data_train_list[max_pres_index] # training data corresponding to best cross-validation accuracy
    max_pres_data_test = data_test_list[max_pres_index] # test data corresponding to best cross-validation accuracy
    max_pres_targets_train = targets_train_list[max_pres_index] # training targets corresponding to best cross-validation accuracy
    max_pres_targets_test = targets_test_list[max_pres_index] # test targets corresponding to best cross-validation accuracy

 

    return mean_precision_score, max_pres_data_train, max_pres_data_test, max_pres_targets_train, max_pres_targets_test

 

def c_matrix(max_pres_data_train, max_pres_data_test, max_pres_targets_train, max_pres_targets_test, tfidf, targets, clf, model_name): #### Creates Confusion matrix for SVC
    tfidf.fit(max_pres_data_train)
    max_pres_data_train_tfidf = tfidf.transform(max_pres_data_train)
    max_pres_data_test_tfidf = tfidf.transform(max_pres_data_test)
    clf.fit(max_pres_data_train_tfidf, max_pres_targets_train) # Fitting SVC
    targets_pred = clf.predict(max_pres_data_test_tfidf) # Prediction on test data
    conf_mat = confusion_matrix(max_pres_targets_test, targets_pred)
    d={'Low':'Negative', 'High':'Positive'}
    rating_df = targets.drop_duplicates().sort_values()
    rating_df= rating_df.apply(lambda x:d[x])
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=rating_df.values, yticklabels=rating_df.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix (Best Precision) - {}".format(model_name))
    plt.show()
def SVC_Save(data, targets, tfidf):
    Path='D:\\Graduate Programs\\Data Mining\\CA Two\\CATwo\\CATwo'
    tfidf.fit(data) # learn vocabulary of entire data
    data_tfidf = tfidf.transform(data)
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary_SVC.csv', header=False)
    print("Shape of tfidf matrix for saved SVC Model: ", data_tfidf.shape)
    clf = LinearSVC().fit(data_tfidf, targets)
    joblib.dump(clf, Path+'svc.sav')
def NBC_Save(data, targets, tfidf):
    Path='D:\\Graduate Programs\\Data Mining\\CA Two\\CATwo\\CATwo'
    tfidf.fit(data) # learn vocabulary of entire data
    data_tfidf = tfidf.transform(data)
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary_NBC.csv', header=False)
    print("Shape of tfidf matrix for saved NBC Model: ", data_tfidf.shape)
    clf = MultinomialNB(alpha=1.0).fit(data_tfidf, targets)
    joblib.dump(clf, Path+'nbc.sav')
    
def main():
    #### Reading training dataset as dataframe
    Path='D:\\Graduate Programs\\Data Mining\\CA Two\\CATwo\\CATwo'
    df = pd.read_csv("MedReviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Normalizing Reviews
    df['Normalized_Review'] = df.Review.apply(normalizer)
    df = df[df['Normalized_Review'].map(len) > 0] # removing rows with normalized reviews of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
    print(df[['Review','Normalized_Review']].head())
    df.drop(['Medicine', 'Condition', 'Review'], axis=1, inplace=True)
    #### Saving cleaned reviews to csv
    df.to_csv(Path+'cleaned_data.csv', encoding='utf-8', index=False)
    #### Reading cleaned reviews as dataframe
    cleaned_data = pd.read_csv(Path+"cleaned_data.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = cleaned_data.Normalized_Review
    targets = cleaned_data.Rating
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=30, norm='l2', ngram_range=(1,3)) # min_df=30 is a clever way of feature engineering

 

    SVC_clf = LinearSVC() # SVC Model
    SVC_mean_precision_score, max_pres_data_train, max_pres_data_test, max_pres_targets_train, max_pres_targets_test = Cross_validation(data, targets, tfidf, SVC_clf, "SVC") # SVC cross-validation
    c_matrix(max_pres_data_train, max_pres_data_test, max_pres_targets_train, max_pres_targets_test, tfidf, targets, SVC_clf, "SVC") # SVC confusion matrix

 

    NBC_clf = MultinomialNB() # NBC Model
    NBC_mean_precision_score, max_pres_data_train, max_pres_data_test, max_pres_targets_train, max_pres_targets_test = Cross_validation(data, targets, tfidf, NBC_clf, "NBC") # NBC cross-validation
    c_matrix(max_pres_data_train, max_pres_data_test, max_pres_targets_train, max_pres_targets_test, tfidf, targets, NBC_clf, "NBC") # NBC confusion matrix

 

    if SVC_mean_precision_score > NBC_mean_precision_score:
        SVC_Save(data, targets, tfidf)
    else:
        NBC_Save(data, targets, tfidf)
if __name__ == "__main__":
    main()
