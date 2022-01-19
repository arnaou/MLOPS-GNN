from ts.torch_handler.base_handler import BaseHandler
import logging
import os
import torch

from src.models.model import GNNModel
from src.features.build_features import process_smiles

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.explain = False
        self.target = 0
        self.model = None

    def load_model(self):
        model_path = "models/checkpoint.pth"
        model = GNNModel.model
        model.load_state_dict(torch.load(model_path))
        self.model = model
        logger.debug(f"Loaded model: {model}")

    def initialize(self):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.load_model()
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data (smiles string) and make it inference ready
        preprocessed_data = process_smiles(data['data'])
        preprocessed_data.batch = torch.tensor([0])
        return preprocessed_data

    def inference(self, data):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        return model_output.item()

    def handle(self, data):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        self.initialize()
        model_input = self.preprocess(data)
        return self.inference(model_input)



if __name__ == "__main__":
    handler = ModelHandler()
    print(handler.handle({'data': 'CCCCCC'}))

