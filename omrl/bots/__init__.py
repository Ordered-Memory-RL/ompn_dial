from .model_bot import *
from .omdec import *


def make(arch, input_dim, action_dim, hidden_size, nb_slots, env_arch, dataloader):
    if arch == 'mlp':
        bot = MLPBot(input_dim=input_dim, action_dim=action_dim,
                     hidden_size=hidden_size, env_arch=env_arch,
                     dataloader=dataloader)

    elif arch == 'lstm':
        bot = LSTMBot(input_dim=input_dim, action_dim=action_dim,
                      hidden_size=hidden_size, num_layers=nb_slots,
                      env_arch=env_arch,
                      dataloader=dataloader)
    elif arch == 'omstack':
        bot = OMStackBot(input_dim=input_dim, action_dim=action_dim,
                         slot_size=hidden_size, nb_slots=nb_slots,
                         env_arch=env_arch, dataloader=dataloader)

    else:
        raise ValueError('No such architecture')
    return bot


