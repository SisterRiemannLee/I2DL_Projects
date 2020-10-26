import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialize h as 0 if these values are not given.                     #
        #######################################################################
        self.linear_x = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(hidden_size, hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bh = torch.zeros(1, hidden_size)
        self.bx = torch.zeros(1, hidden_size)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        seq_len, batch_size, _ = x.size()
        h_seq_augment = torch.zeros(seq_len+1, batch_size, self.hidden_size)
        for i in range(seq_len):
            input = self.linear_x(x[i]) + self.linear_h(h_seq_augment[i]) + self.bh + self.bx
            h_seq_augment[i+1] = torch.tanh(input)

        h_seq = h_seq_augment[1: seq_len+1]
        h = h_seq_augment[seq_len]
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Wc = nn.Linear(input_size, hidden_size)

        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size)

        self.bf = torch.zeros(1, hidden_size)
        self.bi = torch.zeros(1, hidden_size)
        self.bo = torch.zeros(1, hidden_size)
        self.bc = torch.zeros(1, hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        seq_len, batch_size, _ = x.size()

        f_seq = torch.zeros(seq_len, batch_size, self.hidden_size)
        i_seq = torch.zeros(seq_len, batch_size, self.hidden_size)
        o_seq = torch.zeros(seq_len, batch_size, self.hidden_size)

        c_seq_augment = torch.zeros(seq_len + 1, batch_size, self.hidden_size)
        h_seq_augment = torch.zeros(seq_len + 1, batch_size, self.hidden_size)
        for i in range(seq_len):
            f_seq[i] = torch.sigmoid(self.Wf(x[i]) + self.Uf(h_seq_augment[i]) + self.bf)
            i_seq[i] = torch.sigmoid(self.Wi(x[i]) + self.Ui(h_seq_augment[i]) + self.bi)
            o_seq[i] = torch.sigmoid(self.Wo(x[i]) + self.Uo(h_seq_augment[i]) + self.bo)

            c_seq_augment[i + 1] = f_seq[i].mul(c_seq_augment[i]) + i_seq[i].mul(torch.tanh(\
                self.Wc(x[i]) + self.Uc(h_seq_augment[i]) + self.bc))

            h_seq_augment[i + 1] = o_seq[i].mul(torch.tanh(c_seq_augment[i + 1]))

        h_seq = h_seq_augment[1: seq_len + 1]
        c_seq = c_seq_augment[1: seq_len + 1]

        h = h_seq_augment[seq_len]
        c = c_seq_augment[seq_len]
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################
        self.classes = classes
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=activation)
        self.fc = nn.Linear(hidden_size, classes)

    def forward(self, x):
        # h0 = torch.zeros(1, x.size(1), self.hidden_size)

        # Forward propagate RNN
        out, _ = self.rnn(x)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])

        return out
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.classes = classes
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)

    def forward(self, x):

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])

        return out
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
