import re
import collections
import numpy as np
import scipy
import sklearn
import sklearn.feature_extraction.text as sktxt
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn.utils import resample
from sklearn import linear_model as lin
from sklearn.model_selection import cross_val_score,cross_validate,KFold,train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline , make_pipeline
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt

#################################
# loading data  
#################################

def loadTrainfile(filename,maxlines=1000):
    labels = []
    alltxts = []
    with open(filename,"r") as f:
        lines = f.readlines()
        for line in lines[:maxlines]:
            lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",line)
            txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",line)
            alltxts.append(txt)
            if lab.count('M') >0:
                labels.append(-1)définit la taille maximale du
dictionnaire (i.e. le nombre de mots uniques maximal) utilisé pour la vectorisation des données.
            else : 
                labels.append(1)
    return np.array(alltxts),np.array(labels)

def loadTestfile(filename,maxlines=1000):
    alltxts = []
    with open(filename,"r") as f:
        lines = f.readlines()
        for line in lines[:maxlines]:
            txt = re.sub(r"<[0-9]*:[0-9]*>(.*)","\\1",line)
            alltxts.append(txt)
    return np.array(alltxts)


def getScores(model,txts,Y,vectorizer):
    n = 20
    scores = np.zeros(n)
    df_range = np.linspace(1,20,n).astype(int)
    for i,df in enumerate(df_range):
        vectorizer.set_params(min_df = df)
        X = vectorizer.fit_transform(txts)
        scores[i] = cross_val_score( model, X, Y,scoring='f1_macro').mean()
    return df_range,scores


def predictTestFile(model,prediction,filename):
    with open(filename,"w") as f:
        for p in prediction : 
            f.write(f'{p}\n')


def labels_distribution(labels):

    c1 = len(labels[labels == 1])
    c2 = len(labels[labels == -1])
    print(f'Chirac    : {c1}\t{round(c1/len(labels)*100)}%')
    print(f'Mitterand : {c2}\t{round(c2/len(labels)*100)}%')

def analyse_model(model,X,labels):
    x_train, x_test, y_train, y_test = train_test_split(X, labels, random_state= 1) # random state => productibility 
    metrics = ['precision','recall','f1_macro','roc_auc']
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    crossvall_dict = cross_validate (model, x_train, y_train, scoring=metrics)
    for metric in metrics : 
        print(f"CV {metric}:{crossvall_dict['test_'+metric].mean()}" )

    model.fit(x_train,y_train)
    
    c1_test,c1_labels = x_test[y_test == 1], y_test[y_test == 1]
    print(f"Accuracy chirac :{sklearn.metrics.accuracy_score(c1_labels,model.predict(c1_test))}" )
    c2_test,c2_labels = x_test[y_test == -1], y_test[y_test == -1]
    print(f"Accuracy Mitterand :{sklearn.metrics.accuracy_score(c2_labels,model.predict(c2_test))}" )

    c = collections.Counter(model.predict(x_test))
    print(f"prediction count : {c}")

def down_sampling(X,Y):
    X_maj,Y_maj = X[Y==1], Y[Y==1]
    X_min,Y_min = X[Y==-1], Y[Y==-1]
    X_maj_down,Y_maj_down = resample(X_maj,Y_maj,replace=False,n_samples=len(Y_min))

    X_downsampled = np.concatenate([X_min,X_maj_down])
    Y_downsampled = np.concatenate([Y_min,Y_maj_down]) 
    return X_downsampled,Y_downsampled


def ImbalancedTest(alltxts,labels,alltxts_predict):
    # show data characteristics : 
    print("\n\tLabel distribution for DB")
    labels_distribution(labels)

    #######################################
    # Naïve model 
    #######################################
    # vectorizer with no pre-processing : 
    vectorizer = sktxt.CountVectorizer()
    X = vectorizer.fit_transform(alltxts)
    print( "\n\ttest naïve multinomial model : ")
    analyse_model(nb.MultinomialNB(fit_prior=False),X,labels)

    #######################################
    # Upsampling and Down sampling : 
    #######################################

        # Up sampling : 
    model = nb.MultinomialNB(fit_prior=False)
    imba_pipeline = make_pipeline(SMOTE(random_state=42),model)
    print( "\n\ttest with upsampling with SMOTE multinomial model : ")
    analyse_model(imba_pipeline,X,labels)

        # down sampling : 
    X_downsampled,Y_downsampled = down_sampling(alltxts,labels)
    print( "\n\ttest down-sampling  multinomial model : ")
    X = vectorizer.transform(X_downsampled)
    analyse_model(nb.MultinomialNB(fit_prior=False),X,Y_downsampled)

    

def test_Random_Forest(alltxts,labels):
    #######################################
    # Random Forest classifier  : 
    #######################################
    params = {
    'n_estimators': [2, 10, 50,100],
    'max_depth': [2, 4, 8, 16, 32,64],
    'random_state': [13]}
    vectorizer = sktxt.CountVectorizer(min_df = 2,ngram_range=(1,2))
    X,Y = down_sampling(alltxts,labels)
    X = vectorizer.fit_transform(X)

    grid_naive_up = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring='f1_macro').fit(X,Y)
    print(grid_naive_up.cv_results_['mean_test_score'])
    print(grid_naive_up.best_params_)
définit la taille maximale du
dictionnaire (i.e. le nombre de mots uniques maximal) utilisé pour la vectorisation des données.
def pre_processing_test(alltxts,labels):
    alltxts,labels = down_sampling(alltxts,labels)
    
    stop_words = set(stopwords.words('french'))
    stemmer = SnowballStemmer("french") 
    analyzer =sktxt.CountVectorizer().build_analyzer()
    stemming = lambda doc : [stemmer.stem(w) for w in analyzer(doc)]
    stemming_stop_words = lambda doc : [stemmer.stem(w) for w in analyzer(doc) if w not in stop_words]

    vectorizer = sktxt.CountVectorizer(stop_words=stop_words,
                                       ngram_range=(1,2),
                                       tokenizer=stemming)
    model = nb.MultinomialNB(fit_prior=False)
    params = [{},
              {'tokenizer' : stemming},
              {'stop_words': stop_words},
              {'stop_words': stop_words, 'tokenizer' : stemming_stop_words}]
    
    for param in params :
       vectorizer = sktxt.CountVectorizer(ngram_range=(1,2),lowercase = False,**param)
       X = vectorizer.fit_transform(alltxts)
       print('for :',' '.join([f'{k}:{bool(v)}' for k,v in param.items()]))
       analyse_model(model,X,labels)
       words_limits,scores =getScores(model,alltxts,labels,vectorizer)
    
    for param in params :
        vectorizer = sktxt.CountVectorizer(ngram_range=(1,2),lowercase = False,**param)
        words_limits,scores =getScores(model,alltxts,labels,vectorizer)
        plt.title(' '.join([f'{k}:{bool(v)}' for k,v in param.items()]))
        plt.xlabel('nb of documents')
        plt.ylabel('f1_macro')
        plt.plot(words_limits,scores)
        plt.show()

def make_prediction(alltxts,labels,alltxts_predict):
    # Train final algo and print to file : 
        # merge all txt
    stop_words = set(stopwords.words('french'))
    corpus = np.concatenate([alltxts,alltxts_predict])
    
    txts_down,Y = down_sampling(alltxts,labels)
    vectorizer = sktxt.CountVectorizer(ngram_range=(1,2),min_df =2)
    vectorizer.fit(corpus)
    X = vectorizer.transform(txts_down)
    
    X_predict = vectorizer.transform(alltxts_predict)
    model = nb.MultinomialNB(fit_prior=False)
    model.fit(X,Y)
    prediction = model.predict(X_predict)
    predictTestFile(model,prediction,"chirac.pred")

def main():
    print("start")
    nblines = None
    alltxts,labels = loadTrainfile("corpus.tache1.learn.utf8",nblines)
    alltxts_predict = loadTestfile("corpus.tache1.test.utf8",nblines)
    
    print("DB loaded")
    ImbalancedTest(alltxts,labels,alltxts_predict)
    pre_processing_test(alltxts,labels)
    test_Random_Forest(alltxts,labels)
    make_prediction(alltxts,labels,alltxts_predict)


if __name__ == "__main__":
    main()