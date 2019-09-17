import torch
import torch.nn as nn


# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class Actor(nn.Module):

    def __init__(self, n_s, n_a, msg_len, hidden_layers):
        """
        The policy network is allowed to have only one aggregation operation due to communication latency, but we can
        have any number of hidden layers to be executed by each agent individually.
        :param n_s: number of MDP states per agent
        :param n_a: number of MDP actions per agent
        :param hidden_layers: list of ints that will determine the width of each hidden layer
        :param k: aggregation filter length
        :param ind_agg: before which MLP layer index to aggregate
        """
        super(Actor, self).__init__()
        self.msg_len = msg_len
        self.n_s = n_s
        self.n_a = n_a
        self.layers = [n_s + msg_len] + hidden_layers + [n_a + msg_len]
        self.n_layers = len(self.layers) - 1

        self.conv_layers = []

        for i in range(0, self.n_layers):
            m = nn.Conv1d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=1,
                          stride=1)

            self.conv_layers.append(m)

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

    def forward(self, value, network, message):
        """
        The policy relies on delayed information from neighbors. During training, the full history for k time steps is
        necessary.
        :param value:
        :param network:
        :param message:
        :return:
        """
        batch_size = value.shape[0]
        unroll_len = value.shape[1]
        n_agents = value.shape[3]
        assert network.shape[0] == batch_size
        assert network.shape[2] == n_agents
        assert network.shape[3] == n_agents

        assert value.shape[2] == self.n_s

        output_actions = []
        output_messages = []

        current_message = message[:, 0, :, :]

        for l in range(unroll_len):
            passed_messages = torch.matmul(current_message, network[:, l, :, :])  # B, F, N
            current_values = value[:, l, :, :]  # B, F, N
            x = torch.cat((passed_messages, current_values), 1)  # cat in features dim

            for i in range(self.n_layers):
                x = self.conv_layers[i](x)

                if i < self.n_layers - 1:
                    x = torch.tanh(x)
                else:
                    x = 10.0 * torch.tanh(x)

            actions = x[:, 0:self.n_a, :]
            current_message = x[:, self.n_a:self.n_a + self.msg_len, :]

            output_messages.append(current_message.view((batch_size, 1, self.msg_len, n_agents)))
            output_actions.append(actions.view((batch_size, 1, self.n_a, n_agents)))

        return torch.cat(tuple(output_actions), 1), torch.cat(tuple(output_messages), 1)
