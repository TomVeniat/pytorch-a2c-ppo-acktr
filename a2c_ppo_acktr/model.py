import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from supernets.implementation.ThreeDimNeuralFabric import ThreeDimNeuralFabric, Out

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        if 'deter_eval' in base_kwargs:
            self.base = Adaptor(base=base, obs_shape=obs_shape, action_space=action_space, **base_kwargs)
            # self.base = Adaptor(base=base, obs_shape=obs_shape, n_classes=512, **base_kwargs)
            # self.base = ThreeDimCNFAdapter(input_dim=obs_shape, n_classes=action_space.n, **base_kwargs)
            if base == ThreeDimNeuralFabric:
                self.base.critic_linear = nn.Linear(self.base.output_size, 1)

        else:
            self.base = base(obs_shape[0], **base_kwargs)
            self.base.critic_linear = nn.Linear(self.base.output_size, 1)

        if action_space.__class__.__name__ == "Discrete":
            if base == ThreeDimCNFAdapter:
                self.dist = Categorical(None, None, adapt=False)
            else:
                num_outputs = action_space.n
                self.dist = Categorical(self.base.output_size, num_outputs)

        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size, **kwargs):
        super(NNBase, self).__init__(**kwargs)

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())


            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]


            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        # init_ = lambda m: m

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        # init_ = lambda m: m

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

class Adaptor(NNBase):
    def __init__(self, base, recurrent, obs_shape, action_space, static, **kwargs):
        super(Adaptor, self).__init__(recurrent, kwargs['n_classes'], kwargs['n_classes'])
        self.base = base(input_dim=obs_shape, **kwargs)
        self.recurrent = recurrent
        self.static = static
        if not self.static:
            self.gamma = nn.Parameter(torch.ones(1, self.base.n_stoch_nodes)*3)
        else:
            self.gamma = None

    def forward(self, inputs, rnn_hxs, masks):
        self.base.fire(type='new_sequence')
        self.base.log_probas = []

        if self.static:
            probas = torch.ones(1, self.base.n_stoch_nodes).to(inputs.device)
        else:
            probas = self.gamma.sigmoid()

        self.base.set_probas(probas)

        res = self.base(inputs / 255.0)
        if len(res) == 2:
            x, val = res
        else:
            assert len(res) == 1
            x, val = res[0], self.critic_linear(res[0])


        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return val, x, rnn_hxs



class ThreeDimCNFAdapter(ThreeDimNeuralFabric):
    VALUE_OUT_NAME = 'Value'
    PI_OUT_NAME = 'Action'

    def __init__(self, n_classes, critic=True, static=True, hidden_size=None, **args):
        self.hidden_size = hidden_size
        args['n_classes'] = n_classes if hidden_size is None else hidden_size
        super().__init__(**args)

        assert len(self.out_nodes) == 1

        self.static = static
        self.critic = critic

        if self.hidden_size is None:
            value_out = Out(self.n_features, 1, self.bias)

            self.graph.add_node(self.VALUE_OUT_NAME, module=value_out)
            self.register_stochastic_node(self.VALUE_OUT_NAME)

            self.blocks.append(value_out)

            for block in range(self.n_block):
                # Connect all the blocks in last scale, last layer to the Value Out block
                cur_node = (self.n_layer - 1, self.n_scales - 1, block)
                self.graph.add_edge(cur_node, self.VALUE_OUT_NAME, width_node=self.VALUE_OUT_NAME)

            self.set_graph(self.graph, [self.INPUT_NAME], [self.OUTPUT_NAME, self.VALUE_OUT_NAME])

        else:
            value_out = Out(self.hidden_size, 1, self.bias)
            self.graph.add_node(self.VALUE_OUT_NAME, module=value_out)
            self.register_stochastic_node(self.VALUE_OUT_NAME)
            self.blocks.append(value_out)

            self.graph.add_edge(self.OUTPUT_NAME, self.VALUE_OUT_NAME, width_node=self.VALUE_OUT_NAME)

            pi_out = Out(self.hidden_size, n_classes, self.bias)
            self.graph.add_node(self.PI_OUT_NAME, module=pi_out)
            self.register_stochastic_node(self.PI_OUT_NAME)
            self.blocks.append(pi_out)

            self.graph.add_edge(self.OUTPUT_NAME, self.PI_OUT_NAME, width_node=self.PI_OUT_NAME)

            self.set_graph(self.graph, [self.INPUT_NAME], [self.PI_OUT_NAME, self.VALUE_OUT_NAME])


    def forward(self, inputs):
        pi, val = super().forward(inputs)
        return pi, val
