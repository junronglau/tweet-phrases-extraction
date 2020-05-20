# tweet phrases extraction 
This was originally a project submission for Kaggle's Tweet phrase extraction to explore Google's BERT and its variants, as well as a QA approach to extracting relevant text from a document. For learning purposes, it is refactored and deployed as an API using Flask and Docker. You can train your own model and run predictions on it if you have the GPU capabilities.

# Table Of Contents
- [Installation and Usage](#installation-and-usage)
- [Metrics Tracking](#metrics-tracking)
- [Model Architecture](#model-architecture)
- [Project Architecture](#project-architecture)
- [Data Sets](#data-sets)
  - [Train/test data](#train/test-data)
  - [HuggingFace roBERTa base model](#huggingface-roberta-base-model)
- [Acknowledgements](#acknowledgements)

# Installation and Usage
- Install requirements on your virtual environment  

   `pip install -r requirements.txt`

- Update configurations(data/model path) in configs folder

- Run *main.py* to train model from scratch

- Make predictions by running *predict.py*, then the following command:  
  `-curl -X POST -F file=@datasets/test.csv "http://localhost:5000/predict"`

# Metrics Tracking
Add details here on how to track, what can be tracked, screenshots of comet.ml experiments
To monitor and compare different experiments including the code and hyperparameters (think GitHub but for ML models), we can create a Comet.ml experiment. Include the Experiment's API key into configs/roberta_config.json and execute training process as normal.

To specify which parameters(defined in config file) you want to track, modify the init_callbacks method in *roberta_trainer.py*:
```
parameters = {
    "learning_rate" : self.config.model.learning_rate,
    "optimizer" : self.config.model.optimizer,
    "num_epochs": self.config.trainer.num_epochs,
    "batch_size": self.config.trainer.batch_size,
    "validation_split": self.config.trainer.validation_split,
    "verbose_training": self.config.trainer.verbose_training,
}
```
To track additional metrics, modify the predict method in the same file:
```
metrics = {
    'Jaccard similarity': jaccard_score
}
```
These metrics or hyper parameters are then available under their respective tabs
&nbsp;
![](/images/cometml1.png)
&nbsp;  
If you have GitHub linked, you can reproduce the exact commit of a specific Experiment
&nbsp;
![](/images/cometml2.png)

# Model Architecture
The model requires these inputs of a fixed length
- Tokenized documents (These are represented in the form of IDs and sentence seperators)
- Attention mask (These indicates the starting and ending position of our documents)
- Token types (roBERTa does not require Next Sentence Prediction, hence we indicate it as all Ones)
And outputs 2 vectors, one for the starting position and one for ending position. These help us extract the selected_text column which we are predicting.

- roBERTa pretrained model : Similar to BERT(small), this has 12 layers of transformer encoders 
For each head, we have the following layers:
- Drop out layer : 10% drop out rate
- Bidirectional LSTM layer : Using an LSTM layer, we are able to preserve spatial information of the roBERTa output
- Drop out layer : 10% drop out rate
- Dense and activation layer : Condense all nodes and apply the softmax function

# Project Architecture
*main.py* controls the pipeline of the training process. Data will be loaded, tokenized and preprocessed using RobertaDataLoader class. The model is then instantiated with RobertaModel class, and they both of them will be used to train the model using the RobertaTrainer class. A Comet.ml Exeriment will also be created to track the training and testing parameters. 

## Folder Structure
```
├── main.py             - Controls main pipeline for training and saving the model
│           
├── predict.py          - Loads the saved model weights and exposes Flask API for prediction
│
├── requirements.txt    - Stores requirements for project
│   
├── base                        - this folder contains the abstract classes of the project components
│   ├── base_data_loader.py     - this file contains the abstract class of the data loader.
│   ├── base_model.py           - this file contains the abstract class of the model.
│   └── base_train.py           - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the model (roberta model and the fine-tuning layers)
│   └── roberta_model.py
│
│
├── trainer             - this folder contains the trainer responsible for tracking and executing 
|   |                     the training process and predictions
│   └── roberta_trainer.py
│
|
├── data_loader         - this folder contains the data loader responsible for loading and processing the data 
│   └── roberta_data_loader.py
│
│
├── configs             - this folder contains the experiment and model configs including paths to datasets
│   └── roberta_config.json
│
├── datasets            - this folder contains the datasets and roberta model
│   ├── test.csv
|   ├── train.csv
|   └── tf-roberta
│       ├──config-roberta-base.json
│       ├──merges-roberta-base.txt
│       ├──pretrained-roberta-base.h5
│       └──vocab-roberta-base.json
|   
├── notebooks           - this folder contains the exploratory data analysis notebook of the dataset
│
└── utils               - this folder contains util functions
     ├── config.py      - util functions for parsing the config files
     ├── metric.py        - util functions for our Jacaard metric
     └── utils.py       - util functions for parsing arguments
```

# Data Sets
## Train/test data
The dataset contains the text column, which is the original tweet. We are then supposed to predict the selected_text column which is a subset of text column, based on the sentiment given. The format of the sample dataset is shown 
```
| textID     | text                                                                  | selected_text                  | sentiment |
|------------|-----------------------------------------------------------------------|--------------------------------|-----------|
| bdc32ea43c |  Journey!? Wow u just became cooler. is that possible!?               | Wow u just became cooler.      | positive  |
| 088c60f138 | my boss is bullying me...                                             | bullying me                    | negative  |
```
Link to the full dataset can be found from [Kaggle](https://www.kaggle.com/c/tweet-sentiment-extraction/data)

## HuggingFace roBERTa base model 
For this implementation, I used HuggingFace's pretrained roBERTa base model. 
You will need the following files which can be found at [cdeotte's kaggle dataset](https://www.kaggle.com/cdeotte/tf-roberta)

- Config file for the model
- Merges file
- Pretrained model
- Vocab file

Downloading the files is not necessary; You can also initialize your own config or architecture by referring to the [HuggingFace documentation](https://huggingface.co/transformers/model_doc/roberta.html)

# Acknowledgements
Project Template adapted [here](https://github.com/Ahmkel/Keras-Project-Template)  
Data Tokenization and roBERTa model loading adapted [here ](https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705)  
Model architecture inspired from a [Stanford CS224 course project](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15848021.pdf)

