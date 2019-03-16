import torch
import torch.nn as nn
import torch.optim as optim

from a2c_ppo_acktr.algo.kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 path_recorder=None,
                 cost_evaluator=None,
                 arch_loss_coef=0):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.path_recorder = path_recorder
        self.cost_evaluator = cost_evaluator
        self.arch_loss_coef = arch_loss_coef

        # self.optimizer = optim.Adam(
        #         actor_critic.parameters(), lr, eps=eps)
        self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        ### ARCH LOSS
        if self.path_recorder is not None:
            sampled, pruned = self.path_recorder.get_architectures(self.actor_critic.base.base.out_nodes)
            costs_s = self.cost_evaluator.get_costs(sampled)  # Sampled cost
            costs_p = self.cost_evaluator.get_costs(pruned)  # Pruned cost
            stacked_log_probas = torch.stack(self.actor_critic.base.base.log_probas)
            arch_reward = (value_loss * self.value_loss_coef + action_loss) - self.arch_loss_coef * costs_p.mean()
            arch_loss = -(arch_reward * stacked_log_probas).mean()
        else:
            arch_loss = 0
        # print('Sampled={}, pruned={}'.format(costs_s, costs_p))
        ###


        self.optimizer.zero_grad()
        # print('Params: {}'.format(self.actor_critic.base.probas))
        # print('Params: {}'.format(self.actor_critic.base.base.probas))

        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef + arch_loss).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
