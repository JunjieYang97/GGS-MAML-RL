import maml_rl.envs
import gym
import numpy as np
import torch
import json
import torch.nn.utils as utils

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    torch.manual_seed(224552)
    torch.cuda.manual_seed_all(24422)
    np.random.seed(22442)

    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0'])

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device='cuda')
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    # Generate the whole gaussian vector for the whole epoch
    normal_vectors_tensors = []
    for i in range(args.meta_batch_size):
        nvt = []
        for m in range(args.gaus_num):
            normal_vector_tensor = torch.randn(utils.parameters_to_vector(
                policy.parameters()).size()[0]).cuda()
            nvt.append(normal_vector_tensor)
        normal_vectors_tensors.append(nvt)
    
    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    for batch in range(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        if args.ggs_approx:
            episodes, H_norm = metalearner.sample_step_gaus(args.first_order, tasks, normal_vectors_tensors, args.approx_delta, 
                args.gaus_num)
        elif args.zero_order:
            episodes, H_norm = metalearner.zero_order_sample_step(args.first_order, tasks, normal_vectors_tensors, args.approx_delta,
                args.gaus_num)
        else:
            episodes = metalearner.sample(tasks, first_order=args.first_order)
            metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters, 
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                ls_backtrack_ratio=args.ls_backtrack_ratio)

        # Tensorboard
        writer.add_scalar('total_rewards/before_update',
            total_rewards([ep.rewards for ep, _ in episodes]), batch)
        writer.add_scalar('total_rewards/after_update',
            total_rewards([ep.rewards for _, ep in episodes]), batch)
        if args.ggs_approx or args.zero_order:
            writer.add_scalar('Hessian_norm', H_norm, batch)
            print('Hessian_norm', H_norm, batch)

        print('total_rewards/before_update',
            total_rewards([ep.rewards for ep, _ in episodes]), batch)
        print('total_rewards/after_update',
            total_rewards([ep.rewards for _, ep in episodes]), batch)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str, default='2DNavigation-v0',
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.1,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=20,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=8,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    # Gaussian vector approximation
    parser.add_argument('--ggs_approx', action='store_true',
        help='use the gaus vectors approximation of MAML')
    parser.add_argument('--zero_order', action='store_true',
        help='use the function value to evaluate MAML')
    parser.add_argument('--gaus_num', type=int, default=10,
        help='number of gaussian vectors to estimate')
    parser.add_argument('--approx_delta', type=float, default=1e-2,
        help='parameter used in approximation')
    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')

    args.output_folder = args.env_name
    if args.first_order:
        args.output_folder += '_firstorder'
    if args.zero_order:
        args.output_folder += ('_zero_order_gsnum_'+str(args.gaus_num)+'_delta_'+
            str(args.approx_delta)+'_fastlr_'+str(args.fast_lr))
        args.first_order = True
    if args.ggs_approx:
        args.output_folder += ('_ggs_gsnum_'+str(args.gaus_num)+'_delta_'+
            str(args.approx_delta)+'_hidden_'+str(args.hidden_size)+'_layers_'
            +str(args.num_layers)+'_fastlr_'+str(args.fast_lr))
        # have to be first order
        args.first_order = True

    # Device
    args.device = 'cuda'
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
