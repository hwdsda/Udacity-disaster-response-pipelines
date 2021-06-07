# import libraries
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
from sqlalchemy import create_engine

import pickle

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

def load_data(database_filepath):
    """
    Load data from SQL database file
    Assign X and y
    -------

    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('dis_res_table', engine)
    
    X = df['message']
    y = df.iloc[:, 4:]
    
    category_names = y.columns.values
    
    return X, y, category_names


def tokenize(text):
    # url regular expressions
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Find all urls in the provided text
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with url placeholder string
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Extract the word tokens from the provided text
    tokens = word_tokenize(text)
    
    # Init the Wordnet Lemmatizer (converting a word to its base form)
    lemmatizer = WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build a pipeline to process text messages and apply a classifier
    Use grid search to find better paremeters
    """
    # build pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))#,
            
            #('starting_verb_transformer', StartingVerbExtractor())
            
        ])),

        ('clf', MultiOutputClassifier(LogisticRegression(max_iter = 200)))
    ])
    
    # parameters
    parameters = {'clf__estimator__max_iter': [200, 400], 
                  'clf__estimator__C': [0.5, 1]}
    
    # create grid seach object
    pipeline_cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline_cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_test = model.predict(X_test)
    
    #print(classification_report(Y_test.values, y_pred_test, target_names=category_names, zero_division=1))
    
    f1_scores = []
    for ind, cat in enumerate(Y_test):
        print('Class - {}'.format(cat))
        print(classification_report(Y_test.values[ind], y_pred_test[ind], zero_division = 1))
    
        f1_scores.append(f1_score(Y_test.values[ind], y_pred_test[ind], zero_division = 1))
  
    print('Trained Model\nMinimum f1 score - {}\nBest f1 score - {}\nMean f1 score - {}'.format(min(f1_scores), max(f1_scores), round(sum(f1_scores)/len(f1_scores), 3)))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()