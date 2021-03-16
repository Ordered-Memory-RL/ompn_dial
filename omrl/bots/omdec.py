from utils import DictList, get_env_encoder, get_action_dist
from .model_bot import ModelBot
import torch
import torch.nn as nn
from .om_utils import Distribution, ComCell, DecomCell

__all__ = ['OMStackBot']


class OMdecBase(ModelBot):
    """ Base class of OMdec """
    def __init__(self, nb_slots, slot_size, action_dim, input_dim, env_arch, dataloader, dropout=0.,
                 process='stickbreaking'):
        super(OMdecBase, self).__init__(input_dim=input_dim,
                                        hidden_size=slot_size,
                                        env_arch=env_arch,
                                        dataloader=dataloader,
                                        action_dim=action_dim)
        self.nb_slots = nb_slots
        self.slot_size = slot_size
        self.layernorm = nn.LayerNorm(slot_size, elementwise_affine=False)
        self.distribution = Distribution(slot_size * 4, slot_size, dropout=dropout, process=process)
        self.init_p = self.distribution.init_p(1, nslot=self.nb_slots)

        self.actor = nn.Sequential(nn.Linear(3 * slot_size, action_dim))
        self.critic = nn.Sequential(nn.Linear(3 * slot_size, 1))
        self.memory_encoder = get_env_encoder(env_arch, slot_size)

    @property
    def is_recurrent(self):
        return True

    def step(self, obs, task, mems):
        """ Return output, mems, extra_info """
        raise NotImplementedError

    def forward(self, obs, sketchs, sketch_lengths, mems=None) -> DictList:
        task_emb = self.env_emb(sketchs, sketch_lengths)
        obs_inp = self.encode_obs(obs)
        output, memory, extra_info = self.step(obs_inp, task_emb, mems)
        output = torch.cat([output, task_emb, obs_inp], dim=-1)

        # Replace done with p_end
        mean = self.actor(output)
        action_dist = get_action_dist(mean)
        p_end = extra_info['p_end'].clamp(1e-9, 1 - 1e-9)
        results = {'dist': action_dist, 'mems': memory,
                   'log_end': torch.log(p_end),
                   'log_no_end': torch.log(1 - p_end)}
        results.update(extra_info)
        return DictList(results)

    def teacherforcing_batch(self, batch: DictList, batch_lengths, sketch_lengths,
                             recurrence) -> (DictList, DictList):
        bsz, seqlen = batch.actions.shape[0], batch.actions.shape[1]
        #normalized_actions = self.anorm.normalize(batch.actions)
        sketchs = batch.tasks
        task_emb = self.env_emb(sketchs, sketch_lengths)
        obs_repr = self.encode_obs(batch.states)
        extra_infos = DictList({})
        mems = self.init_memory(sketchs, sketch_lengths)
        cell_outputs = []
        for t in range(seqlen):
            obs_inp = obs_repr[:, t]
            final_output = DictList({})
            cell_output, next_mems, extra_info = self.step(obs_inp, task_emb, mems)
            cell_outputs.append(cell_output)
            extra_infos.append(extra_info)

            # Update memory
            if (t + 1) % recurrence == 0:
                next_mems = next_mems.detach()
            mems = next_mems

        cell_outputs = torch.stack(cell_outputs, dim=1)
        extra_infos.apply(lambda _t: torch.stack(_t, dim=1))
        outputs = torch.cat([cell_outputs, task_emb[:, None, :].repeat(1, seqlen, 1), obs_repr], dim=-1)
        means = self.actor(outputs)
        action_dist = get_action_dist(means)
        logprobs = action_dist.log_prob(batch.actions.float())

        p_end = extra_infos.p_end.clamp(1e-20, 1 - 1e-20)
        log_end = torch.log(p_end)
        log_no_end = torch.log(1 - p_end)
        log_no_end_term = log_no_end + logprobs

        # The log probs are action log probs until last one
        final_logprobs = log_no_end_term
        batch_ids = torch.arange(bsz, device=batch.states.device)
        final_logprobs[batch_ids, batch_lengths - 1] = log_end[batch_ids, batch_lengths - 1]

        # Final outputs
        final_outputs = DictList({})
        final_outputs.loss = -final_logprobs
        sequence_mask = torch.arange(batch_lengths.max().item(),
                                     device=batch_lengths.device)[None, :] < batch_lengths[:, None]
        final_outputs.apply(lambda _t: _t.masked_fill(~sequence_mask, 0.))
        return final_outputs, extra_infos


class OMStackBot(OMdecBase):
    def __init__(self, input_dim, action_dim, slot_size, env_arch, dataloader, nb_slots=3, dropout=0,
                 process='stickbreaking'):
        super(OMStackBot, self).__init__(input_dim=input_dim, action_dim=action_dim, slot_size=slot_size,
                                         nb_slots=nb_slots, dropout=dropout, process=process,
                                         env_arch=env_arch, dataloader=dataloader)
        self.com_cell = nn.ModuleList(
            [ComCell(hidden_size=slot_size, dropout=dropout) for _ in range(nb_slots)])
        self.decom_cell = nn.ModuleList(
            [DecomCell(hidden_size=slot_size, dropout=dropout) for _ in range(nb_slots)])

    def step(self, input_enc, task_emb, memory):
        prev_m, prev_p = self._unflat_memory(memory)
        bsz, nslot, _ = prev_m.size()
        comb_input = torch.cat([input_enc, task_emb], dim=-1)

        p_hat = self.init_p.repeat([bsz, 1]).to(device=input_enc.device)
        cand_m = prev_m
        not_init_id = torch.nonzero((prev_p.sum(-1) != 0)).squeeze(1)
        if len(not_init_id) > 0:
            cm_list = []
            selected_inp = comb_input[not_init_id]
            selected_prev_m = prev_m[not_init_id]
            h = input_enc[not_init_id]
            for i in range(self.nb_slots - 1, -1, -1):
                h = self.com_cell[i](h, selected_prev_m[:, i, :], selected_inp)
                cm_list.append(h)
            selected_cand_m = torch.stack(cm_list[::-1], dim=1)
            cand_m[not_init_id] = selected_cand_m

            dist_input = torch.cat([selected_inp[:, None, :].expand(-1, nslot, -1),
                                    selected_prev_m,
                                    selected_cand_m], dim=-1)
            p_hat[not_init_id] = self.distribution(dist_input)

        p_end = p_hat[:, 0]
        p = torch.nn.functional.normalize(p_hat[:, 1:], dim=1, p=1)
        cp = p.cumsum(dim=1)
        rcp = p.flip([1]).cumsum(dim=1).flip([1])

        chl = torch.zeros_like(cand_m[:, 0])
        chl_list = []
        for i in range(self.nb_slots):
            chl_list.append(chl)
            h = rcp[:, i, None] * cand_m[:, i] + (1 - rcp)[:, i, None] * chl
            chl = self.decom_cell[i](parent=h, inp_enc=comb_input, context=None)
        chl_array = torch.stack(chl_list, dim=1)

        m = prev_m * (1 - cp)[:, :, None] + cand_m * p[:, :, None] + chl_array * (1 - rcp)[:, :, None]
        output = m[:, -1]
        return output, self._flat_memory(m, p), {'p': p_hat, 'p_end': p_end}

    def _flat_memory(self, mem, p):
        batch_size = mem.shape[0]
        mem_size = self.nb_slots * self.slot_size
        return torch.cat([mem.reshape(batch_size, mem_size),
                          p.reshape(batch_size, self.nb_slots)], dim=1)

    def _unflat_memory(self, memory):
        mem_size = self.nb_slots * self.slot_size
        mem = memory[:, :mem_size].reshape(-1, self.nb_slots, self.slot_size)
        p = memory[:, mem_size:]
        return mem, p

    def init_memory(self, sketchs, sketch_lengths):
        batch_size = sketchs.shape[0]
        device = sketchs.device
        first_slot = self.layernorm(self.memory_encoder(sketchs.long(), sketch_lengths.long()))
        init_m = nn.functional.pad(first_slot[:, None, :], [0, 0, 0, self.nb_slots - 1], value=0.)
        init_p = torch.zeros([batch_size, self.nb_slots], device=device)
        return torch.cat([init_m.reshape(batch_size, -1),
                          init_p], dim=-1)

