"""
Modular policy
https://github.com/KyriacosShiarli/taco
"""
from torch import nn
import torch
from utils import DictList, get_action_dist, Normaliser


class ForwardModel(nn.Module):
    def __init__(self, in_dim=(39+40+40+1), final_dim=39, num_units=[300, 200, 100]):
        super(ForwardModel, self).__init__()
        actor_layers = []
        for out_dim in num_units:
            actor_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        # Final layers
        actor_layers.append(nn.Linear(in_dim, final_dim))
        self.actor_layers = nn.ModuleList(actor_layers)
        self.depth = len(num_units)
        self.selu = nn.SELU(inplace=True)

    def forward(self, inputs, dropout_rate=0.0):
        actor_output = inputs
        for i in range(self.depth):
            actor_output = self.selu(self.actor_layers[i](actor_output))

        actor_output = self.actor_layers[-1](actor_output)
        return actor_output


# class ModularPolicy(nn.Module):
#     def __init__(
#         self,
#         nb_subtasks,
#         input_dim,
#         n_actions,
#         a_mu=None,
#         a_std=None,
#         s_mu=None,
#         s_std=None,
#     ):
#         super(ModularPolicy, self).__init__()
#         self.mlps = nn.ModuleList(
#             [HMLP(in_dim=input_dim, n_actions=n_actions) for _ in range(nb_subtasks)]
#         )

#         self.candidates = None
#         self.sketch_id = None

#         if a_mu is None or a_std is None:
#             a_mu = torch.zeros(n_actions)
#             a_std = torch.ones(n_actions)
#         if s_mu is None or s_std is None:
#             s_mu = torch.zeros(input_dim)
#             s_std = torch.ones(input_dim)
#         self.anorm = Normaliser(a_mu, a_std)
#         self.snorm = Normaliser(s_mu, s_std)

#     def reset(self, subtasks):
#         self.sketch_id = 0
#         self.candidates = [self.mlps[subtask] for subtask in subtasks]

#     def get_action(self, inputs, mode="greedy", sketch_idx=None):
#         action = None
#         normalized = self.snorm.normalize(inputs)
#         if sketch_idx is not None:
#             self.sketch_id = sketch_idx
#         while True:
#             if self.sketch_id == len(self.candidates):
#                 break

#             curr_mlp = self.candidates[self.sketch_id]
#             action_outputs = curr_mlp.get_action(normalized, mode)
#             stop = action_outputs.stop.item()
#             if stop == 1:
#                 self.sketch_id += 1
#             else:
#                 action = self.anorm.recover(action_outputs.action)
#                 break

#         return action

#     def forward(self, task, states, actions, dropout_p=0.0):
#         """
#         Args:
#             task:  int
#             states:  [bsz, in_dim]
#             actions: [bsz, a_dim]

#         Returns:
#             action_logprobs [bsz]
#             stop_logprobs [bsz, 2]
#         """
#         normalized_states = self.snorm.normalize(states)
#         normalized_actions = self.anorm.normalize(actions)
#         mlp = self.mlps[task]
#         model_output = mlp(normalized_states, dropout_p)
#         action_logprobs = model_output.action_dist.log_prob(normalized_actions)
#         stop_logprobs = torch.nn.functional.log_softmax(
#             model_output.stop_dist.logits, dim=-1
#         )
#         return {"action_logprobs": action_logprobs, "stop_logprobs": stop_logprobs}
