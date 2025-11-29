import pandas as pd
import random
import logging
import argparse
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import csv
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer


import sys
sys.path.append('.')

from scorer.task import evaluate, evaluate_1C
from format_checker.task import check_format

random.seed(1234)
ROOT_DIR = dirname(dirname(__file__))
# print("Root:", ROOT_DIR)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def read_csv(data_fpath, sep):
    data = {}
    with open(data_fpath) as f:
        reader = csv.reader(f, delimiter=sep)
        header = next(reader)
        for row in reader:
            for i in range(len(header)):
                if header[i] not in data:
                    data[header[i]] = []
                data[header[i]].append(row[i])
    return data

def run_majority_baseline(data_fpath, test_fpath, results_fpath, subtask):
    # train_df = pd.read_csv(data_fpath, dtype=object, encoding="utf-8", sep='\t')
    # test_df = pd.read_csv(test_fpath, dtype=object, encoding="utf-8", sep='\t')
    train_df = read_csv(data_fpath, sep='\t')
    test_df = read_csv(test_fpath, sep='\t')

    if subtask == "1A" or subtask == "1B":
        pipeline = DummyClassifier(strategy="most_frequent")

        text_head = "text"
        id_head = "id"

        pipeline.fit(train_df[text_head], train_df['label'])
        
        with open(results_fpath, "w") as results_file:
            predicted_distance = pipeline.predict(test_df[text_head])

            results_file.write("id\tlabel\tmodel\n")

            for i, line in enumerate(test_df[text_head]):
                label = predicted_distance[i]

                results_file.write("{}\t{}\t{}\n".format(test_df[id_head][i], label, 'majority baseline'))
    else:
        pipeline1 = DummyClassifier(strategy="most_frequent")
        pipeline2 = DummyClassifier(strategy="most_frequent")
        pipeline3 = DummyClassifier(strategy="most_frequent")

        text_head = "text"
        id_head = "id"
        pipeline1.fit(train_df[text_head], train_df['hate_type'])
        pipeline2.fit(train_df[text_head], train_df['hate_severity'])
        pipeline3.fit(train_df[text_head], train_df['to_whom'])
        with open(results_fpath, "w") as results_file:
            predicted_distance1 = pipeline1.predict(test_df[text_head])
            predicted_distance2 = pipeline2.predict(test_df[text_head])
            predicted_distance3 = pipeline3.predict(test_df[text_head])
            results_file.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
            for i, line in enumerate(test_df[text_head]):
                label1 = predicted_distance1[i]
                label2 = predicted_distance2[i]
                label3 = predicted_distance3[i]

                results_file.write("{}\t{}\t{}\t{}\t{}\n".format(test_df[id_head][i], label1, label2, label3, 'majority baseline'))


def run_random_baseline(data_fpath, results_fpath, subtask):
    gold_df = read_csv(data_fpath, sep='\t')
    #label_list=gold_df['label'].to_list()
    hate_labels = ["Abusive", "Political Hate", "Profane", "Religious Hate", "Sexism", "None"]
    to_whom_labels = ["Society", "Organization", "Community", "Individual", "None"]
    severity_labels = ["Little to None", "Mild", "Severe"]

    if subtask == "1A":
        id_head = "id"
        with open(results_fpath, "w") as results_file:
            results_file.write("id\tlabel\tmodel\n")
            for i, line in enumerate(gold_df[id_head]):
                results_file.write('{}\t{}\t{}\n'.format(line, random.choice(hate_labels), 'random baseline'))
    elif subtask == "1B":
        id_head = "id"
        with open(results_fpath, "w") as results_file:
            results_file.write("id\tlabel\tmodel\n")
            for i, line in enumerate(gold_df[id_head]):
                results_file.write('{}\t{}\t{}\n'.format(line, random.choice(to_whom_labels), 'random baseline'))
    else:
        id_head = "id"
        with open(results_fpath, "w") as results_file:
            results_file.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
            for i, line in enumerate(gold_df[id_head]):
                results_file.write('{}\t{}\t{}\t{}\t{}\n'.format(line, random.choice(hate_labels),
                                                         random.choice(severity_labels),
                                                         random.choice(to_whom_labels),
                                                         'random baseline'))


def run_ngram_baseline(train_fpath, test_fpath, results_fpath, subtask):
    # train_df = pd.read_csv(train_fpath, dtype=object, encoding="utf-8", sep='\t')
    # test_df = pd.read_csv(test_fpath, dtype=object, encoding="utf-8", sep='\t')
    train_df = read_csv(train_fpath, sep='\t')
    test_df = read_csv(test_fpath, sep='\t')

    if subtask == "1A" or subtask == "1B":
        text_head = "text"
        id_head = "id"

        pipeline = Pipeline([
            ('ngrams', TfidfVectorizer(ngram_range=(1, 1),lowercase=True,use_idf=True,max_df=0.95, min_df=3,max_features=5000)),
            ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
        ])
        pipeline.fit(train_df[text_head], train_df['label'])

        with open(results_fpath, "w") as results_file:
            predicted_distance = pipeline.predict(test_df[text_head])
            results_file.write("id\tlabel\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label = predicted_distance[i]
                results_file.write("{}\t{}\t{}\n".format(line, label, 'n-gram'))
    else:
        text_head = "text"
        id_head = "id"

        pipeline1 = Pipeline([
            ('ngrams', TfidfVectorizer(ngram_range=(1, 1), lowercase=True, use_idf=True, max_df=0.95, min_df=3,
                                       max_features=5000)),
            ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
        ])
        pipeline1.fit(train_df[text_head], train_df['hate_type'])
        pipeline2 = Pipeline([
            ('ngrams', TfidfVectorizer(ngram_range=(1, 1), lowercase=True, use_idf=True, max_df=0.95, min_df=3,
                                       max_features=5000)),
            ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
        ])
        pipeline2.fit(train_df[text_head], train_df['hate_severity'])
        pipeline3 = Pipeline([
            ('ngrams', TfidfVectorizer(ngram_range=(1, 1), lowercase=True, use_idf=True, max_df=0.95, min_df=3,
                                       max_features=5000)),
            ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
        ])
        pipeline3.fit(train_df[text_head], train_df['to_whom'])

        with open(results_fpath, "w") as results_file:
            predicted_distance1 = pipeline1.predict(test_df[text_head])
            predicted_distance2 = pipeline2.predict(test_df[text_head])
            predicted_distance3 = pipeline3.predict(test_df[text_head])
            results_file.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label1 = predicted_distance1[i]
                label2 = predicted_distance2[i]
                label3 = predicted_distance3[i]

                results_file.write("{}\t{}\t{}\t{}\t{}\n".format(line, label1, label2, label3, 'n-gram'))




def run_logreg_baseline(train_fpath, test_fpath, results_fpath, subtask):
    train_df = read_csv(train_fpath, sep='\t')
    test_df = read_csv(test_fpath, sep='\t')

    text_head = "text"
    id_head = "id"

    # Define the simple CountVectorizer for basic bag-of-words (unigrams)
    vectorizer = CountVectorizer(ngram_range=(1, 1), lowercase=True, max_df=0.95, min_df=3, max_features=5000)

    if subtask == "1A" or subtask == "1B":
        pipeline = Pipeline([
            ('count_vec', vectorizer),
            ('clf', LogisticRegression(random_state=0, solver='liblinear', C=1.0))
        ])
        pipeline.fit(train_df[text_head], train_df['label'])

        with open(results_fpath, "w") as results_file:
            predicted_distance = pipeline.predict(test_df[text_head])
            results_file.write("id\tlabel\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label = predicted_distance[i]
                results_file.write("{}\t{}\t{}\n".format(line, label, 'logreg-count'))
    else:
        # Subtask 1C (Multi-task)
        pipeline1 = Pipeline([('count_vec', vectorizer), ('clf', LogisticRegression(random_state=0, solver='liblinear', C=1.0))])
        pipeline1.fit(train_df[text_head], train_df['hate_type'])
        
        pipeline2 = Pipeline([('count_vec', vectorizer), ('clf', LogisticRegression(random_state=0, solver='liblinear', C=1.0))])
        pipeline2.fit(train_df[text_head], train_df['hate_severity'])
        
        pipeline3 = Pipeline([('count_vec', vectorizer), ('clf', LogisticRegression(random_state=0, solver='liblinear', C=1.0))])
        pipeline3.fit(train_df[text_head], train_df['to_whom'])

        with open(results_fpath, "w") as results_file:
            predicted_distance1 = pipeline1.predict(test_df[text_head])
            predicted_distance2 = pipeline2.predict(test_df[text_head])
            predicted_distance3 = pipeline3.predict(test_df[text_head])
            results_file.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label1 = predicted_distance1[i]
                label2 = predicted_distance2[i]
                label3 = predicted_distance3[i]

                results_file.write("{}\t{}\t{}\t{}\t{}\n".format(line, label1, label2, label3, 'logreg'))


def run_randomforest_baseline(train_fpath, test_fpath, results_fpath, subtask):
    train_df = read_csv(train_fpath, sep='\t')
    test_df = read_csv(test_fpath, sep='\t')

    text_head = "text"
    id_head = "id"

    vectorizer = CountVectorizer(ngram_range=(1, 1), lowercase=True, max_df=0.95, min_df=3, max_features=5000)

    if subtask == "1A" or subtask == "1B":
        pipeline = Pipeline([
            ('count_vec', vectorizer),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=0))
        ])
        pipeline.fit(train_df[text_head], train_df['label'])

        with open(results_fpath, "w") as results_file:
            predicted_distance = pipeline.predict(test_df[text_head])
            results_file.write("id\tlabel\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label = predicted_distance[i]
                results_file.write("{}\t{}\t{}\n".format(line, label, 'randomforest-count'))
    else:
        # Subtask 1C (Multi-task)
        pipeline1 = Pipeline([('count_vec', vectorizer), ('clf', RandomForestClassifier(n_estimators=100, random_state=0))])
        pipeline1.fit(train_df[text_head], train_df['hate_type'])
        
        pipeline2 = Pipeline([('count_vec', vectorizer), ('clf', RandomForestClassifier(n_estimators=100, random_state=0))])
        pipeline2.fit(train_df[text_head], train_df['hate_severity'])
        
        pipeline3 = Pipeline([('count_vec', vectorizer), ('clf', RandomForestClassifier(n_estimators=100, random_state=0))])
        pipeline3.fit(train_df[text_head], train_df['to_whom'])

        with open(results_fpath, "w") as results_file:
            predicted_distance1 = pipeline1.predict(test_df[text_head])
            predicted_distance2 = pipeline2.predict(test_df[text_head])
            predicted_distance3 = pipeline3.predict(test_df[text_head])
            results_file.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label1 = predicted_distance1[i]
                label2 = predicted_distance2[i]
                label3 = predicted_distance3[i]

                results_file.write("{}\t{}\t{}\t{}\t{}\n".format(line, label1, label2, label3, 'randomforest'))


def run_decisiontree_baseline(train_fpath, test_fpath, results_fpath, subtask):
    train_df = read_csv(train_fpath, sep='\t')
    test_df = read_csv(test_fpath, sep='\t')

    text_head = "text"
    id_head = "id"

    vectorizer = CountVectorizer(ngram_range=(1, 1), lowercase=True, max_df=0.95, min_df=3, max_features=5000)

    if subtask == "1A" or subtask == "1B":
        pipeline = Pipeline([
            ('count_vec', vectorizer),
            ('clf', DecisionTreeClassifier(random_state=0))
        ])
        pipeline.fit(train_df[text_head], train_df['label'])

        with open(results_fpath, "w") as results_file:
            predicted_distance = pipeline.predict(test_df[text_head])
            results_file.write("id\tlabel\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label = predicted_distance[i]
                results_file.write("{}\t{}\t{}\n".format(line, label, 'dtree-count'))
    else:
        # Subtask 1C (Multi-task)
        pipeline1 = Pipeline([('count_vec', vectorizer), ('clf', DecisionTreeClassifier(random_state=0))])
        pipeline1.fit(train_df[text_head], train_df['hate_type'])
        
        pipeline2 = Pipeline([('count_vec', vectorizer), ('clf', DecisionTreeClassifier(random_state=0))])
        pipeline2.fit(train_df[text_head], train_df['hate_severity'])
        
        pipeline3 = Pipeline([('count_vec', vectorizer), ('clf', DecisionTreeClassifier(random_state=0))])
        pipeline3.fit(train_df[text_head], train_df['to_whom'])

        with open(results_fpath, "w") as results_file:
            predicted_distance1 = pipeline1.predict(test_df[text_head])
            predicted_distance2 = pipeline2.predict(test_df[text_head])
            predicted_distance3 = pipeline3.predict(test_df[text_head])
            results_file.write("id\thate_type\thate_severity\tto_whom\tmodel\n")
            for i, line in enumerate(test_df[id_head]):
                label1 = predicted_distance1[i]
                label2 = predicted_distance2[i]
                label3 = predicted_distance3[i]

                results_file.write("{}\t{}\t{}\t{}\t{}\n".format(line, label1, label2, label3, 'dtree'))


def run_baselines(train_fpath, test_fpath, subtask):
    
    majority_baseline_fpath = join(ROOT_DIR,
                                 f'baselines/prediction/majority_baseline_subtask_{subtask}_{basename(test_fpath)}')
    run_majority_baseline(train_fpath, test_fpath, majority_baseline_fpath, subtask)
    if check_format(majority_baseline_fpath, subtask):
        if subtask == "1A" or subtask == "1B":
            acc, precision, recall, f1 = evaluate(majority_baseline_fpath, test_fpath, subtask)
        else:
            acc, precision, recall, f1 = evaluate_1C(majority_baseline_fpath, test_fpath)
        # logging.info(f"Majority Baseline F1-micro: {f1}")
        logging.info("Majority Baseline: Accuracy={:.4f}\tPrecision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))



    random_baseline_fpath = join(ROOT_DIR, f'baselines/prediction/random_baseline_subtask_{subtask}_{basename(test_fpath)}')
    run_random_baseline(test_fpath, random_baseline_fpath, subtask)
    if check_format(random_baseline_fpath, subtask):
        if subtask == "1A" or subtask == "1B":
            acc, precision, recall, f1 = evaluate(random_baseline_fpath, test_fpath, subtask)
        else:
            acc, precision, recall, f1 = evaluate_1C(random_baseline_fpath, test_fpath)
        # logging.info(f"Random Baseline F1-micro: {f1}")
        logging.info("Random Baseline: Accuracy={:.4f}\tPrecision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))


    ngram_baseline_fpath = join(ROOT_DIR, f'baselines/prediction/ngram_baseline_subtask_{subtask}_{basename(test_fpath)}')
    run_ngram_baseline(train_fpath, test_fpath, ngram_baseline_fpath, subtask)
    if check_format(ngram_baseline_fpath, subtask):
        if subtask == "1A" or subtask == "1B":
            acc, precision, recall, f1 = evaluate(ngram_baseline_fpath, test_fpath, subtask)
        else:
            acc, precision, recall, f1 = evaluate_1C(ngram_baseline_fpath, test_fpath)
        # logging.info(f"N-gram Baseline F1-micro: {f1}")
        logging.info("N-gram Baseline: Accuracy={:.4f}\tPrecision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))

    logreg_baseline_fpath = join(ROOT_DIR, f'baselines/prediction/logreg_baseline_subtask_{subtask}_{basename(test_fpath)}')
    run_logreg_baseline(train_fpath, test_fpath, logreg_baseline_fpath, subtask)
    if check_format(logreg_baseline_fpath, subtask):
        if subtask == "1A" or subtask == "1B":
            acc, precision, recall, f1 = evaluate(logreg_baseline_fpath, test_fpath, subtask)
        else:
            acc, precision, recall, f1 = evaluate_1C(logreg_baseline_fpath, test_fpath)
        logging.info("Logistic Regression Baseline: Accuracy={:.4f}\tPrecision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))


    randomforest_baseline_fpath = join(ROOT_DIR, f'baselines/prediction/randomforest_baseline_subtask_{subtask}_{basename(test_fpath)}')
    run_randomforest_baseline(train_fpath, test_fpath, randomforest_baseline_fpath, subtask)
    if check_format(randomforest_baseline_fpath, subtask):
        if subtask == "1A" or subtask == "1B":
            acc, precision, recall, f1 = evaluate(randomforest_baseline_fpath, test_fpath, subtask)
        else:
            acc, precision, recall, f1 = evaluate_1C(randomforest_baseline_fpath, test_fpath)
        logging.info("Random Forest: Accuracy={:.4f}\tPrecision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))
        

    decisiontree_baseline_fpath = join(ROOT_DIR, f'baselines/prediction/decisiontree_baseline_subtask_{subtask}_{basename(test_fpath)}')
    run_decisiontree_baseline(train_fpath, test_fpath, decisiontree_baseline_fpath, subtask)
    if check_format(decisiontree_baseline_fpath, subtask):
        if subtask == "1A" or subtask == "1B":
            acc, precision, recall, f1 = evaluate(decisiontree_baseline_fpath, test_fpath, subtask)
        else:
            acc, precision, recall, f1 = evaluate_1C(decisiontree_baseline_fpath, test_fpath)
        logging.info("Dicision Tree: Accuracy={:.4f}\tPrecision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(acc, precision, recall, f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", "-t", required=True, type=str,
                        help="The absolute path to the training data")
    parser.add_argument("--dev-file-path", "-d", required=True, type=str,
                        help="The absolute path to the dev data")
    parser.add_argument("--subtask", "-s", required=True, type=str,
                       choices=['1A', '1B', '1C'],
                       help="the subtask")

    args = parser.parse_args()
    run_baselines(args.train_file_path, args.dev_file_path, args.subtask)


