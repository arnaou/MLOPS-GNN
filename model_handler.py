from ts.torch_handler.base_handler import BaseHandler
import logging
import os
import torch

from src.models.model import GNNModel

logger = logging.getLogger(__name__)

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.model = None
        
    
        
    def load_model(self):
        model_path = "model/checkpoint.pth"
        model = GNNModel()
        model.load_state_dict(torch.load(model_path))
        self.model = model
        logger.debug(f"Loaded model: {model}")
        

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.load_model()
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        return preprocessed_data


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model(model_input)
        return model_output


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        return self.inference(model_input) 
        
