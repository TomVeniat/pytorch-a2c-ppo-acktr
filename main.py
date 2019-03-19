from datetime import datetime

import copy
import glob
import os
import time
import socket
from collections import deque

import numpy as np
import torch
from supernets.implementation.ComputationCostEvaluator import ComputationCostEvaluator
from supernets.implementation.ThreeDimNeuralFabric import ThreeDimNeuralFabric
from supernets.interface.PathRecorder import PathRecorder
from tqdm import tqdm

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, ThreeDimCNFAdapter
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    args.log_dir = os.path.join(args.log_dir, socket.gethostname(), datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom

        if args.use_cnf:
            env = 'v3_cnf_l{}_b{}_s{}_c{}_h{}_{}'.format(args.nlayer, args.nblock, args.nscale, args.nchan, args.hidden, "nobn" if args.nobn else "bn")
        elif args.use_cnf_full:
            env = 'v3_cnfull_l{}_b{}_s{}_c{}_h{}_{}'.format(args.nlayer, args.nblock, args.nscale, args.nchan, args.hidden, "nobn" if args.nobn else "bn")

        else:
            env = 'v3_base_mine_fixed'
        env_url = f"http://{args.server}:{args.port}/env/{env}"
        viz = Visdom(server=args.server, port=args.port, env=env)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device, False)
    if args.use_cnf or args.use_cnf_full:
        base_kwargs = {
            'n_layer': args.nlayer,
            'n_block': args.nblock,
            'n_scale': args.nscale,
            'n_chan': args.nchan,

            'deter_eval': True,

            'recurrent': args.recurrent_policy,
            'static': args.static,

            'bn': not args.nobn,
        }
        if args.use_cnf:
            base_kwargs['n_classes'] = args.hidden
            base = ThreeDimNeuralFabric
        else:
            base_kwargs['hidden_size'] = args.hidden
            # base_kwargs['n_classes'] = args.hidden
            base_kwargs['n_classes'] = envs.action_space.n

            base = ThreeDimCNFAdapter
    else:
        base_kwargs = {'recurrent': args.recurrent_policy}
        base = None

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, base=base,
                          base_kwargs=base_kwargs)
    actor_critic.to(device)
    if args.use_cnf:
        path_recorder = PathRecorder(actor_critic.base.base)
        cost_evaluator = ComputationCostEvaluator(node_index=path_recorder.node_index, bw=False)
        cost_evaluator.init_costs(actor_critic.base.base)
        print('Cost: {:.5E}'.format(cost_evaluator.total_cost))
        print(actor_critic)
        print(actor_critic.base.base)
    else:
        path_recorder = None
        cost_evaluator = None


    agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               path_recorder=path_recorder, cost_evaluator=cost_evaluator,
                               arch_loss_coef=args.arch_loss_coef)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    print(actor_critic)

    start = time.time()
    for j in tqdm(range(num_updates), desc='Updates'):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_lr_decay:
            agent.clip_param = args.clip_param * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            tqdm.write("Updates {}, num timesteps {}, FPS {} ({})\n"
                  "\tLast {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       args.log_dir,
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
            if args.vis:
                tqdm.write('\t{}'.format(env_url))
            # print('Params: {}'.format(actor_critic.base.gamma))

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                       actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                           for done_ in done],
                                           dtype=torch.float32,
                                           device=device)
                
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                  format(len(eval_episode_rewards),
                         np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_env_steps)


if __name__ == "__main__":
    main()
