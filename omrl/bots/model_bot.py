import torch
import torch.nn as nn
from utils import get_env_encoder, Normaliser

__all__ = ['MLPBot', 'LSTMBot', 'ModelBot']

from utils import DictList, get_action_dist


class ModelBot(torch.nn.Module):
    def __init__(self, input_dim, hidden_size,
                 action_dim, env_arch,
                 dataloader=None):
        super(ModelBot, self).__init__()
        self.input_dim = input_dim
        self.obs_size = hidden_size

        # Image encoder
        self.inp_enc = nn.Linear(self.input_dim, hidden_size)

        if dataloader is None:
            a_mu = torch.zeros(action_dim)
            a_std = torch.ones(action_dim)
            s_mu = torch.zeros(input_dim)
            s_std = torch.ones(input_dim)
        self.anorm = Normaliser(dataloader.a_mu, dataloader.a_std)
        self.snorm = Normaliser(dataloader.s_mu, dataloader.s_std)

        # Sketch encoder
        self.env_emb = get_env_encoder(env_arch, hidden_size)

    def forward(self, obs, sketchs, sketch_lengths, mems=None) -> DictList:
        """ Single step forward (Imitate babyAI)
        :param obs: [bsz, input_dim]
        :param sketchs: [bsz, task]
        :param sketch_lengths: [bsz]
        :param mems: [bsz, mem_size] or None
        :return DictList with keys "dist", "v", "mems" (optional)
        """
        raise NotImplementedError

    def encode_obs(self, obs: DictList):
        return self.inp_enc(self.snorm.normalize(obs.float()))

    def encode_sketch(self, sketchs, sketch_lengths):
        return self.env_emb(sketchs, sketch_lengths)

    def get_action(self, obs: DictList, sketchs, sketch_lengths, mems) -> DictList:
        """
        obs: [bsz, obs_size]
        mode: "greedy" | "sample"
        """
        output = self.forward(obs, sketchs, sketch_lengths, mems)
        dist = output.dist
        output.actions = dist.mean
        return output

    @property
    def is_recurrent(self):
        return False

    def init_memory(self, sketchs, sketch_lengths):
        """ Return initial memory """
        return None

    def teacherforcing_batch(self, batch: DictList, batch_lengths, sketch_lengths,
                             recurrence) -> (DictList, DictList):
        """
        :param batch: DictList object [bsz, seqlen]
        :param batch_lengths: [bsz]
        :param sketch_lengths: [bsz]
        :param recurrence: an int
        :return:
            stats: A DictList of bsz, mem_size
            extra_info: A DictList of extra info
        """
        bsz, seqlen = batch.actions.shape[0], batch.actions.shape[1]
        sketchs = batch.tasks
        final_outputs = DictList({})
        extra_info = DictList({})
        mems = None
        if self.is_recurrent:
            mems = self.init_memory(sketchs, sketch_lengths)

        for t in range(seqlen):
            final_output = DictList({})
            model_output = self.forward(batch.states[:, t], sketchs, sketch_lengths, mems)
            logprobs = model_output.dist.log_prob(batch.actions[:, t].float())
            if 'log_end' in model_output:
                # p_end + (1 - pend) action_prob
                log_no_end_term = model_output.log_no_end + logprobs
                logprobs = torch.logsumexp(torch.stack([model_output.log_end, log_no_end_term], dim=-1), dim=-1)
                final_output.log_end = model_output.log_end
            final_output.logprobs = logprobs
            if 'p' in model_output:
                extra_info.append({'p': model_output.p})
            final_outputs.append(final_output)

            # Update memory
            next_mems = None
            if self.is_recurrent:
                next_mems = model_output.mems
                if (t + 1) % recurrence == 0:
                    next_mems = next_mems.detach()
            mems = next_mems

        # Stack on time dim
        final_outputs.apply(lambda _tensors: torch.stack(_tensors, dim=1))
        extra_info.apply(lambda _tensors: torch.stack(_tensors, dim=1))
        sequence_mask = torch.arange(batch_lengths.max().item(),
                                     device=batch_lengths.device)[None, :] < batch_lengths[:, None]
        final_outputs.loss = -final_outputs.logprobs
        if 'log_end' in final_outputs:
            batch_ids = torch.arange(bsz, device=batch.states.device)
            final_outputs.loss[batch_ids, batch_lengths - 1] = final_outputs.log_end[batch_ids, batch_lengths - 1]
        final_outputs.apply(lambda _t: _t.masked_fill(~sequence_mask, 0.))
        return final_outputs, extra_info


class MLPBot(ModelBot):
    def __init__(self, input_dim, action_dim, hidden_size, env_arch, dataloader):
        super(MLPBot, self).__init__(input_dim=input_dim, hidden_size=hidden_size, env_arch=env_arch,
                                     action_dim=action_dim, dataloader=dataloader)
        self.actor = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, action_dim))

    def forward(self, obs, sketchs, sketch_lengths, mems=None) -> DictList:
        obs_repr = self.encode_obs(obs)
        env_emb = self.encode_sketch(sketchs, sketch_lengths)
        inp = torch.cat([obs_repr, env_emb], dim=-1)
        mean = self.actor(inp)
        dist = get_action_dist(mean)
        return DictList({'dist': dist})


class LSTMBot(ModelBot):
    def __init__(self, input_dim, action_dim, hidden_size, env_arch, num_layers, dataloader):
        super(LSTMBot, self).__init__(input_dim=input_dim,
                                      hidden_size=hidden_size,
                                      env_arch=env_arch,
                                      action_dim=action_dim,
                                      dataloader=dataloader)
        self.actor = nn.Sequential(nn.Linear(3 * hidden_size, action_dim))
        self.lstm = torch.nn.LSTM(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mem_size = 2*num_layers*hidden_size
        self.memory_encoder = get_env_encoder(env_arch, mem_size)

    @property
    def is_recurrent(self):
        return True

    def _flat_mem(self, lstm_mems):
        # [num_layers, bsz, hidden_size, 2]
        mems = torch.stack(lstm_mems, -1)

        # [bsz, num_layers, hidden_size, 2]
        mems = mems.permute(1, 0, 2, 3)
        return mems.reshape(mems.shape[0], -1)

    def _unflat_mem(self, mems):
        """
        :param mems: [bsz, mem_size]
        :return:
        """
        # bsz, num_layers, hidden_size, 2
        mems = mems.view(mems.shape[0], self.lstm.num_layers, self.lstm.hidden_size, 2)
        mems = mems.permute(1, 0, 2, 3)
        lstm_mems = mems.chunk(2, dim=-1)
        return lstm_mems[0].squeeze(-1).contiguous(), lstm_mems[1].squeeze(-1).contiguous()

    def init_memory(self, sketchs, sketchs_emb):
        """
        :param env_ids: [bsz]
        :return: init_mem: [bsz, mem_size]
        """
        return self.memory_encoder(sketchs, sketchs_emb)

    def forward(self, obs, sketchs, sketch_lengths, mems=None) -> DictList:
        """
        :param obss: [bsz, obs_size]
        :param mems: [bsz, mem_size]
        :return:
        """
        inputs = self.layernorm(self.encode_obs(obs))
        lstm_mems = self._unflat_mem(mems)
        outputs, next_lstm_mems = self.lstm(inputs.unsqueeze(1), lstm_mems)
        next_mems = self._flat_mem(next_lstm_mems)
        outputs = outputs.squeeze(1)
        outputs = torch.cat([outputs, self.env_emb(sketchs, sketch_lengths), inputs], dim=-1)
        mean = self.actor(outputs)
        results = {'mems': next_mems,
                   'dist': get_action_dist(mean)}
        return DictList(results)
