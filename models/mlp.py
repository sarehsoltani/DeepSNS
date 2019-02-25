import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import config


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        
        # specify a simple multilayer perceptron
        self.layer_1 = nn.Linear(in_features=config.NUM_CLASSIFIER_FEATURES,
                                out_features=config.NUM_L1_UNITS)
                                
        self.layer_2 = nn.Linear(in_features=config.NUM_L1_UNITS,
                                out_features=config.NUM_L2_UNITS)
        init.xavier_normal(self.layer_2.weight)
        init.constant(self.layer_2.bias, 0.1)

        self.layer_3 = nn.Linear(in_features=config.NUM_L2_UNITS,
                                out_features=config.NUM_CLASSES)
        init.xavier_normal(self.layer_3.weight)
        init.constant(self.layer_3.bias, 0.1)

    def forward(self, x):
        
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        # x = F.softmax(x, dim=1)

        return x



