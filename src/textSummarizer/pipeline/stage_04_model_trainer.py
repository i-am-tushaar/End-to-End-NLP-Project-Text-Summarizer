from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
            load_instructions = ConfigurationManager()
            model_training_instructions = load_instructions.get_model_trainer_config()
            model_training_manager = ModelTrainer(config=model_training_instructions)
            model_training_manager.train()