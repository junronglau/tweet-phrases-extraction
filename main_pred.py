from comet_ml import Experiment
from data_loader.roberta_data_loader import RobertaDataLoader
from models.roberta_model import RobertaModel
from trainers.roberta_trainer import RobertaTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import flask

app = flask.Flask(__name__)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    global data_loader
    global trainer
    global experiment

    print('Create the data generator.')
    data_loader = RobertaDataLoader(config)

    print('Create the model.')
    model = RobertaModel(config)
    model.load_weights() 

    print('Creating the Experiment')
    experiment = Experiment(api_key=config.exp.comet_api_key, project_name=config.exp.name, auto_output_logging="simple")
    
    print('Create the trainer')
    trainer = RobertaTrainer(model.model, experiment, config) 

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get('file'):
            csv_file = flask.request.files['file']
            with experiment.test():
                print('Predicting the testing data')
                X, y = data_loader.load_data(csv_file)
                X, _ = data_loader.preprocess_data(X, y)
                pred_text = trainer.predict([X, y], data_loader.get_tokenizer())
                data['predictions'] = pred_text
                data['success'] = True
    return flask.jsonify(data)

if __name__ == '__main__':
    print("Loading Keras model and Flask starting server...")
    main()
    app.run(host='0.0.0.0')





