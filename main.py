from comet_ml import Experiment
from data_loader.roberta_data_loader import RobertaDataLoader
from models.roberta_model import RobertaModel
from trainers.roberta_trainer import RobertaTrainer
from utils.config import process_config
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    print('Create the data generator.')
    data_loader = RobertaDataLoader(config)

    print('Create the model.')
    model = RobertaModel(config)

    print('Creating the Experiment')
    experiment = Experiment(api_key=config.exp.comet_api_key, project_name=config.exp.name, auto_output_logging="simple")
    
    print('Create the trainer')
    trainer = RobertaTrainer(model.model, experiment, config, data_loader.get_train_data())
    
    with experiment.train():
        print('Start training the model.')
        trainer.train()
        model.save()

    with experiment.test():
        print('Predicting the testing data')
        trainer.predict(data_loader.get_test_data(), data_loader.get_tokenizer())
    
if __name__ == '__main__':
    main()





