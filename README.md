# tweet phrases extraction

This was originally a project submission for Kaggle's Tweet phrase extraction but for learning purposes, it is refactored and deployed as an API using Flask and Docker

- Install requirements on your virtual env with requirements.txt
- Run main.py to train model from scratch
- Or load model using pretrained weights and serve as an API using Flask
- Added metrics to track training of model using Comet.ml (add your comet key in configurations)

To run the app to predict data, first run main_pred.py, then the following cmd:
`-curl -X POST -F file=@test.csv "http://localhost:5000/predict"`

# Credits
Project Template adapted from: https://github.com/Ahmkel/Keras-Project-Template
