# [Hatespeech Identification Shared Task](https://multihate.github.io/) at [BLP Workshop @IJCNLP-AACL 2025](https://blp-workshop.github.io/)



## Objective

The [Bangla Multi-task Hate Speech Identification shared task](https://multihate.github.io/) is designed to address the complex and nuanced problem of detecting and understanding hate speech in Bangla across multiple related subtasks such as type of hate, severity, and target group. Find the  [Task Description](#task-description) below.


__Table of contents:__
- [Contents of the Directory](#contents-of-the-directory)
- [File Structure](#file_structure)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [Baseline Script and Official Evaluation Metrics](#baseline_script-and-official-evaluation-metrics)
- [Baselines](#baselines)
- [Running Transformer Models](#running-transformer-models)
- [Organizers](#organizers)



## Contents of the Directory
* Main folder: [data](./data)<br/>
  This directory contains data files for the task.
* Main folder: [scripts](./scripts)<br/>
    Contains scripts provided to run transformer-based models for subtask 1A and subtask 1B. 
* Main folder: [output-subtask-1a](./output-subtask-1a)<br/>
    Contains an output files genrated from the run for subtask 1A.
* Main folder: [output-subtask-1a](./format_checker)<br/>
    Contains an output files genrated from the run for subtask 1B.
* [baseline.ipynb](./baseline.ipynb) <br/>
    Driver code to run baseline scripts and generate baseline results. 
* [blp-subtask-1a.ipynb](./blp-subtask-1a.ipynb) <br/>
    Driver codce to run scripts for transformer models for subtask 1A. 
* [blp-subtask-1b.ipynb](./blp-subtask-1b.ipynb) <br/>
    Driver codce to run scripts for transformer models for subtask 1B. 
* [README.md](./README.md) <br/>
    This file!

## File Structure 
Hate_Speech_Classification/
│
├── baseline.ipynb
├── blp-subtask-1a.ipynb
├── blp-subtask-1b.ipynb
│
├── data/
│   ├── sub-task-1a/
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│   │   ├── original_data/
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   ├── tokenized/
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   └── train.csv
│   ├── sub-task-1b/
|   ├── sub-task-1c/
│
├── output-subtask-1a/
│   ├── output_banglabert/
│   ├── output_bert-base-multilingual-cased/
│   ├── output_distilbert-base-cased/
│   ├── output_distilbert-base-uncased/
│   └── output_xlm-roberta-base/
│
├── output-subtask-1b/
│   ├── output_banglabert/
│   ├── output_bert-base-multilingual-cased/
│   ├── output_distilbert-base-cased/
│   ├── output_distilbert-base-uncased/
│   └── xlm-roberta-base/
│
├── scripts/
│   ├── baselines/
│   │   ├── format_checker/
│   │   │   └── task.py
│   │   ├── prediction/
│   │   |   └── baseline_prediction_files
│   │   └── scorer/
│   │       └── task.py
│   │
│   ├── task.py
│   ├── run_glue_v1.py
│   └── run_glue_v2.py
│
└── README.md
├── requirements.txt



## Task Description

This shared task is designed to identify the type of hate, its severity, and the targeted group from social media content. The goal is to develop robust systems that advance research in this area. This shared task have three subtasks:

- **Subtask 1A**: Given a Bangla text collected from YouTube comments, categorize whether it contains _Abusive_, _Sexism_, _Religious Hate_, _Political Hate_, _Profane_, or _None_.
- **Subtask 1B**: Given a Bangla text collected from YouTube comments, categorize whether the hate towards _Individuals_, _Organizations_, _Communities_, or _Society_.
- **Subtask 1C**: This subtask is a multi-task setup. Given a Bangla text collected from YouTube comments, categorize it into type of hate, severity, and targeted group.

**We only focus on subtask 1A and 1B for this project.**

## Dataset
For a brief overview of the dataset, kindly refer to the *README.md* file located in the data directory.


### Input data format

#### Subtask 1A
Each file uses the tsv format. A row within the tsv adheres to the following structure:

```
id	text	label
```
Where:
* id: an index or id of the text
* text: text
* label: _Abusive_, _Sexism_, _Religious Hate_, _Political Hate_, _Profane_, or _None_.

##### Example
```
490273	আওয়ামী লীগের সন্ত্রাসী কবে দরবেন এই সাহস আপনাদের নাই	Political Hate
```

#### Subtask 1B
Each file uses the tsv format. A row within the tsv adheres to the following structure:

```
id	text	label
```
Where:
* id: an index or id of the text
* text: text
* label: _Individuals_, _Organizations_, _Communities_, or _Society_.

##### Example
```
490273	আওয়ামী লীগের সন্ত্রাসী কবে দরবেন এই সাহস আপনাদের নাই	Organization
```

#### Subtask 1C
Each file uses the tsv format. A row within the tsv adheres to the following structure:

```
id	text	hate_type   hate_severity   to_whom
```
Where:
* id: an index or id of the text
* text: text
* hate_type: _Abusive_, _Sexism_, _Religious Hate_, _Political Hate_, _Profane_, or _None_.
* hate_severity: _Little to None_, _Mild_, or _Severe_.
* to_whom: _Individuals_, _Organizations_, _Communities_, or _Society_.

##### Example
```
490273	আওয়ামী লীগের সন্ত্রাসী কবে দরবেন এই সাহস আপনাদের নাই	"Political Hate"  "Little to None"  Organization
```


## Baseline Script and Official Evaluation Metrics

### Baseline Script
The scorer for the task is located in the [scripts/baselines](./scripts/baselines) module of the project. The scorer reports official evaluation metrics and other metrics of a prediction file. 


You can install all prerequisites through,
```
pip install -r requirements.txt
```
Launch the scorer for the task as follows:
```
python scripts/baselines/task.py \
--train-file-path=<train_file> \
--test-file-path=<test_file> \
-- subtask = <subtask 1A, 1B, or 1C>
```


##### Example

```
#For subtask 1A
!python scripts/baselines/task.py \
  --train-file-path data/sub-task-1a/train.tsv \
  --dev-file-path data/sub-task-1a/dev.tsv \
  --subtask 1A
```

**Alternatively running baseline.ipynb would produce the basline results**


## Running Transformer Models 

The files (blp-subtask-1a.ipynb and blp-subtask-1b.ipynb) provide details for running the scripts/run_glue_v2.py. A sample command for running the script is provided in the following: 

```
!python scripts/run_glue_v2.py \
  --model_name_or_path distilbert-base-cased \
  --train_file ./data/sub-task-1a/tokenized/train.csv \
  --validation_file ./data/sub-task-1a/tokenized/dev.csv \
  --test_file ./data/sub-task-1a/tokenized/test.csv \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --output_dir ./output-subtask-1a/output_distilbert-base-cased/ \
  --overwrite_output_dir
```

The above example shows the parameters for running distilbert. Besides distilbet, we also ran m-Bert-base, xlmROBETa-base, and banlgaBERT for both the subtasks. To run other models model name needs to be changed an appropriate output directory should be used to safely output files.  


### Official Evaluation Metrics
The **official evaluation metric** for the subtask 1A and 1B is **micro-F1**. 


## Baselines

The [baselines](baselines) module currently contains a majority, random and a simple n-gram Support Vector Machine (SVM) baseline.  For this project, we have downsampled the data by one-third to reduce computational cost and time. Using the downsampled data, we have reproduced the baseline scores. Besides the mentioned baselines, we also incorporate Logistic Regression, Random Forest, and Decision Tree as baseline methods.


#### Subtask 1A

Baseline Results for the task on Test set (Evaluation Phase)

| Model                      | micro-F1 |
|----------------------------|----------|
| Random Baseline            | 0.1609   |
| Majority Baseline          | 0.5703   |
| n-gram (SVM) Baseline      | 0.6079   |
| Logistic Regression        | 0.6041   |
| Random Forest              | 0.5779   |
| Decision Tree              | 0.4812   |




Baseline Results for the task on Dev-Test set

| Model                      | micro-F1 |
|----------------------------|----------|
| Random Baseline            | 0.1398   |
| Majority Baseline          | 0.5639   |
| n-gram (SVM) Baseline      | 0.5974   |
| Logistic Regression        | 0.5926   |
| Random Forest              | 0.5878   |
| Decision Tree              | 0.5161   |


#### Subtask 1B
Baseline Results for the task on Test set (Evaluation Phase)

| Model                      | micro-F1 |
|----------------------------|----------|
| Random Baseline            | 0.2082  |
| Majority Baseline          | 0.6038   |
| n-gram (SVM) Baseline      | 0.6250   |
| Logistic Regression        | 0.6215   |
| Random Forest              | 0.6003   |
| Decision Tree              | 0.4782   |


Baseline Results for the task on Dev-Test set

| Model                      | micro-F1 |
|----------------------------|----------|
| Random Baseline            | 0.2222   |
| Majority Baseline          | 0.5747   |
| n-gram (SVM) Baseline      | 0.6057   |
| Logistic Regression        | 0.6129   |
| Random Forest              | 0.5926   |
| Decision Tree              | 0.4970   |







<!-- **Note that the checker cannot verify whether the prediction file you submit contains all lines, because it does not have access to the corresponding gold file.** -->


## Project Collaborator
- [Krishno Dey](https://krishnodey.github.io/), PhD Student, University of New Brunswick
- [Yalda Keivan Jafari](https:///), MCS Student, University of New Brunswick

This project is conducted as final project for the CS6765: Natural Language Processing, at University of New Brunswick.  





## Credit goes to the task organizers
- [Md Arid Hasan](https:aridhasan.github.io), PhD Student, The University of Toronto
- [Firoj Alam](https://firojalam.one/), Senior Scientist, Qatar Computing Research Institute
- Md Fahad Hossain, Lecturer, Daffodil International University
- [Usman Naseem](https://usmaann.github.io/), Assistant Professor, Macquarie University
- [Syed Ishtiaque Ahmed](https://www.ishtiaque.net/), Associate Professor, The University of Toronto
