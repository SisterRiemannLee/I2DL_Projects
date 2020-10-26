import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)

        self.dense1 = nn.Linear(6400, 2000)
        self.dense2 = nn.Linear(2000, 500)
        self.dense3 = nn.Linear(500, 30)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.uniform_(m.weight, a=0, b=0.01)
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform_(m.weight, gain=1)
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        x = self.pool(F.elu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.elu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.elu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.elu(self.conv4(x)))
        x = self.drop4(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.elu(self.dense1(x))  # Dense, Activation
        x = self.drop5(x)

        x = torch.tanh(self.dense2(x))  # Dense, Activation
        x = self.drop6(x)

        x = self.dense3(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
