from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer as \
    torch_normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np

torch, nn = try_import_torch()


class SKUStoreBatchNormModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization."""
    capture_index = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in [16, 8]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU))
            prev_layer_size = size
            # Add a batch norm layer.
            # layers.append(nn.BatchNorm1d(prev_layer_size))

        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=self.num_outputs,
            initializer=torch_normc_initializer(0.01),
            activation_fn=None)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=torch_normc_initializer(1.0),
            activation_fn=None)

        self._hidden_layers = nn.Sequential(*layers)
        self._hidden_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=input_dict.get("is_training", False))
        self._hidden_out = self._hidden_layers(input_dict["obs"])
        logits = self._logits(self._hidden_out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])


class SKUWarehouseBatchNormModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization."""
    capture_index = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in [16, 8]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU))
            prev_layer_size = size
            # Add a batch norm layer.
            # layers.append(nn.BatchNorm1d(prev_layer_size))

        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=self.num_outputs,
            initializer=torch_normc_initializer(0.01),
            activation_fn=None)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=torch_normc_initializer(1.0),
            activation_fn=None)

        self._hidden_layers = nn.Sequential(*layers)
        self._hidden_out = None
    

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=input_dict.get("is_training", False))
        self._hidden_out = self._hidden_layers(input_dict["obs"])
        logits = self._logits(self._hidden_out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])



class SKUStoreDNN(FullyConnectedNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.obs_size = obs_space.shape[0]
        self.hidden_dim = 8
        self._hidden_layers = nn.Sequential(nn.Linear(self.obs_size, 2*self.hidden_dim), 
                              nn.ReLU(), 
                              nn.Linear(2*self.hidden_dim, self.hidden_dim))
        self._logits = nn.Sequential(nn.ReLU(), nn.Linear(self.hidden_dim, num_outputs))
        self.value_branch = nn.Linear(self.hidden_dim, 1)
        self._cur_value = None
    
    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
    
    @override(FullyConnectedNetwork)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        assert torch.sum(torch.isnan(obs)).item() == 0, "obs"
        assert torch.sum(torch.isinf(obs)).item() == 0, "obs"
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        assert torch.sum(torch.isnan(self._features)).item() == 0, 'features'
        assert torch.sum(torch.isinf(self._features)).item() == 0, 'features'
        logits = self._logits(self._features) if self._logits else \
            self._features
        assert torch.sum(torch.isnan(logits)).item() == 0, 'logits'
        assert torch.sum(torch.isinf(logits)).item() == 0, 'logits'
        self._cur_value = self.value_branch(self._features).squeeze(1)
        assert torch.sum(torch.isnan(self._cur_value)).item() == 0, 'cur_value'
        assert torch.sum(torch.isinf(self._cur_value)).item() == 0, 'cur_value'

        assert torch.sum(torch.isnan(list(self.parameters())[0].data)).item() == 0, 'weights'
        assert torch.sum(torch.isinf(list(self.parameters())[0].data)).item() == 0, 'weights inf'
        return logits, state
    

class SKUStoreGRU(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.obs_size = obs_space.shape[0]
        self.max_seq_len = model_config['max_seq_len']
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.fc1 = nn.Sequential(nn.Linear(self.obs_size, 2*self.rnn_hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(2*self.rnn_hidden_dim, self.rnn_hidden_dim))
        self.rnn = nn.GRU(self.rnn_hidden_dim, self.rnn_hidden_dim, 1)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(self.rnn_hidden_dim, num_outputs))
        self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
        self._cur_value = None
    
    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [torch.randn(1, self.rnn_hidden_dim).squeeze(0)]
        return h
    
    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
    
    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
            inputs (dict): Observation tensor with shape [B, T, obs_size].
            state (list): List of state tensors, each with shape [B, size].
            seq_lens (Tensor): 1D tensor holding input sequence lengths.
                Note: len(seq_lens) == B.
        """
        x = self.fc1(inputs).permute(1, 0, 2)
        h_in = state[-1]
        h_in = h_in.reshape(1, h_in.size(0), h_in.size(1)) 
        output, hidden_state = self.rnn(x, h_in)
        new_state = hidden_state.reshape(hidden_state.size(1), -1)
        q = self.fc2(output)
        self._cur_value = self.value_branch(output).reshape([-1])
        return q, [new_state]


class SKUWarehouseNet(RecurrentNetwork, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.obs_size = obs_space.shape[0]
        self.max_seq_len = model_config['max_seq_len']
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.fc1 = nn.Sequential([nn.Linear(self.obs_size, 2*self.rnn_hidden_dim), 
                                  nn.ReLU(), 
                                  nn.Linear(2*self.rnn_hidden_dim, self.rnn_hidden_dim)])
        self.rnn = nn.GRU(self.rnn_hidden_dim, self.rnn_hidden_dim, 1)
        self.fc2 = nn.Sequential([nn.ReLU(), nn.Linear(self.rnn_hidden_dim, num_outputs)])
        self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
        self._cur_value = None
    
    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [self.fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)]
        return h
    
    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
    
    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
            inputs (dict): Observation tensor with shape [B, T, obs_size].
            state (list): List of state tensors, each with shape [B, size].
            seq_lens (Tensor): 1D tensor holding input sequence lengths.
                Note: len(seq_lens) == B.
        """
        
        x = self.fc1(inputs).permute(1, 0, 2)
        h_in = state[-1]
        h_in = h_in.reshape(1, h_in.size(0), h_in.size(1)) 
        output, hidden_state = self.rnn(x, h_in)
        new_state = hidden_state.reshape(hidden_state.size(1), -1)
        q = self.fc2(output)
        self._cur_value = self.value_branch(output).reshape([-1])
        return q, [new_state]