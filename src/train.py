"""
Training script to train a model for semantic textual similarity task on STS Benchmark dataset.
"""
import importlib
import util.parameters as params

FIXED_PARAMETERS = params.load_parameters()
model_type = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model_type])) 
STSModel = getattr(module, 'STSModel')

# Parameter settings at each launch of training script
print("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)

class modelClassifier:
    def __init__(self, seq_length):
        # Define hyperparameters
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.max_len = FIXED_PARAMETERS["seq_length"]

        print("Building model from %s.py" %(model_type))
        self.model_type = STSModel(max_len=self.max_len, emb_train=self.emb_train)

classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])
