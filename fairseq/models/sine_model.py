from . import BaseFairseqModel, register_model, register_model_architecture
from torch import nn


@register_model('sine_model')
class SineModel(BaseFairseqModel):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def max_positions(self):
        """Maximum length supported by the model."""
        return 1

    def __call__(self, x):
        return self.model(x)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        model = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1),)
        return SineModel(model=model)


# default parameters used in tensor2tensor implementation
@register_model_architecture('sine_model', 'sine_model')
def sine_model_architecture(args):
    pass
