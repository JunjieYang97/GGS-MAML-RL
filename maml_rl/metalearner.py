import torch
import numpy as np
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
from copy import deepcopy

class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

    def inner_update(self, grad_diff_matrix, normal_matrix, grad_final_tensor, U):
        # parameter update based on Hessian vector and grad vectors
        grad_right = torch.unsqueeze(grad_final_tensor, 1)
        H_u = torch.matmul(torch.stack(normal_matrix), grad_right)
        H_u = torch.matmul(grad_diff_matrix, H_u)
        H_u = torch.div(H_u, U)

        H_matrix = torch.matmul(grad_diff_matrix, torch.stack(normal_matrix))
        H_matrix = torch.div(H_matrix, U)
        H_matrix_norm = torch.norm(H_matrix).detach().cpu().numpy()
        grad_right = grad_right - self.fast_lr*H_u
        para_update = torch.squeeze(grad_right)
        return para_update, H_matrix_norm

    def inner_update_zero(self, loss_diff_list, normal_matrix, grad_final_tensor, U):
        grad_right = torch.unsqueeze(grad_final_tensor, 1)
        H_u = torch.matmul(torch.stack(normal_matrix), grad_right)
        temp = torch.ones(normal_matrix[0].size()[0], 1).cuda()
        loss_diff_matrix = torch.matmul(temp, loss_diff_list)
        loss_normal = torch.t(torch.stack(normal_matrix))*loss_diff_matrix
        H_u = torch.matmul(loss_normal, H_u)
        H_u = torch.div(H_u, U)

        H_matrix = torch.matmul(loss_normal, torch.stack(normal_matrix))
        H_matrix = torch.div(H_matrix, U)
        H_matrix_norm = torch.norm(H_matrix).detach().cpu().numpy()
        grad_right = grad_right - self.fast_lr*H_u
        para_update = torch.squeeze(grad_right)
        return para_update, H_matrix_norm

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0,
            weights=episodes.mask)

        return loss

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr,
            first_order=first_order)

        return params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                        gamma=self.gamma, device=self.device)
            params = self.adapt(train_episodes, first_order=first_order)
            valid_episodes = self.sampler.sample(self.policy, params=params,
                        gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def sample_step_gaus(self, first_order, tasks, nv_tensor_matrix, delta, U, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        old_pis, old_losses, episodes_list = [], [], []
        para_update_list, H_matrix_norm_list = [], []
        for (task, nv_tensor_list) in zip(tasks, nv_tensor_matrix):
            self.sampler.reset_task(task)
            grads_u = []
            for u in range(U+2):
                if u >= U:
                    train_episodes = self.sampler.sample(self.policy,
                        gamma=self.gamma, device=self.device)
                    params = self.adapt(train_episodes, first_order=first_order)
                    valid_episodes = self.sampler.sample(self.policy, params=params,
                        gamma=self.gamma, device=self.device)
                    old_loss, _, old_pi = self.surrogate_loss_gaus((train_episodes, valid_episodes))
                    grads = torch.autograd.grad(old_loss, self.policy.parameters())
                    grads = parameters_to_vector(grads)
                    grads_u.append(grads)
                    if u > U:
                        old_pis.append(old_pi)
                        old_losses.append(old_loss)
                        episodes_list.append((train_episodes, valid_episodes))
                else:
                    para_vector = parameters_to_vector(self.policy.parameters())
                    vector_to_parameters((para_vector+delta*nv_tensor_list[u]), 
                        self.policy.parameters())
                    train_episodes = self.sampler.sample(self.policy,
                        gamma=self.gamma, device=self.device)
                    params = self.adapt(train_episodes, first_order=first_order)
                    valid_episodes = self.sampler.sample(self.policy, params=params,
                    gamma=self.gamma, device=self.device)
                    old_loss, _, old_pi = self.surrogate_loss_gaus((train_episodes, valid_episodes))
                    grads = torch.autograd.grad(old_loss, self.policy.parameters())
                    grads = parameters_to_vector(grads)
                    grads_u.append(grads)
                    vector_to_parameters(para_vector, self.policy.parameters())

            grad_original = torch.unsqueeze(grads_u[-2], 1)
            grad_diff_matrix = torch.t(torch.stack(grads_u[0:-2])) - torch.matmul(grad_original, torch.ones(1, U).cuda())
            grad_diff_matrix = torch.div(grad_diff_matrix, delta)
            para_update, H_matrix_norm = self.inner_update(grad_diff_matrix, nv_tensor_list, grads_u[-1], U)
            para_update_list.append(para_update)
            H_matrix_norm_list.append(H_matrix_norm)

        grads_trpo = torch.mean(torch.stack(para_update_list), dim=0)
        old_losses = torch.mean(torch.stack(old_losses, dim=0))
        H_matrix_norm_mean = np.mean(H_matrix_norm_list)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes_list,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads_trpo,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier
        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes_list, old_pis=old_pis)
            improve = loss - old_losses
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())
        return episodes_list, H_matrix_norm_mean

    def zero_order_sample_step(self, first_order, tasks, nv_tensor_matrix, delta, U, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        old_pis, old_losses, episodes_list = [], [], []
        para_update_list, H_matrix_norm_list = [], []
        for (task, nv_tensor_list) in zip(tasks, nv_tensor_matrix):
            self.sampler.reset_task(task)
            loss_add_list, loss_minus_list = [], []
            for u in range(U+2):
                if u >= U:
                    train_episodes = self.sampler.sample(self.policy,
                        gamma=self.gamma, device=self.device)
                    params = self.adapt(train_episodes, first_order=first_order)
                    valid_episodes = self.sampler.sample(self.policy, params=params,
                        gamma=self.gamma, device=self.device)
                    old_loss, _, old_pi = self.surrogate_loss_gaus((train_episodes, valid_episodes))
                    if u == U:
                        grads = torch.autograd.grad(old_loss, self.policy.parameters())
                        grads = parameters_to_vector(grads)
                    else:
                        old_pis.append(old_pi)
                        loss_original = old_loss
                        old_losses.append(old_loss)
                        episodes_list.append((train_episodes, valid_episodes))
                else:
                    para_vector = parameters_to_vector(self.policy.parameters())
                    vector_to_parameters((para_vector+delta*nv_tensor_list[u]), 
                        self.policy.parameters())
                    train_episodes = self.sampler.sample(self.policy,
                        gamma=self.gamma, device=self.device)
                    params = self.adapt(train_episodes, first_order=first_order)
                    valid_episodes = self.sampler.sample(self.policy, params=params,
                    gamma=self.gamma, device=self.device)
                    old_loss, _, old_pi = self.surrogate_loss_gaus((train_episodes, valid_episodes))                    
                    loss_add_list.append(old_loss)
                    vector_to_parameters((para_vector-delta*nv_tensor_list[u]), 
                        self.policy.parameters())
                    train_episodes = self.sampler.sample(self.policy,
                        gamma=self.gamma, device=self.device)
                    params = self.adapt(train_episodes, first_order=first_order)
                    valid_episodes = self.sampler.sample(self.policy, params=params,
                    gamma=self.gamma, device=self.device)
                    old_loss, _, old_pi = self.surrogate_loss_gaus((train_episodes, valid_episodes))
                    loss_minus_list.append(old_loss)
                    vector_to_parameters(para_vector, self.policy.parameters())
            
            loss_diff_matrix = torch.t(torch.stack(loss_add_list)) + torch.t(torch.stack(loss_minus_list)) \
                - 2*loss_original*torch.ones(1, U).cuda()
            loss_diff_matrix = torch.div(loss_diff_matrix, 2*delta*delta)
            para_update, H_matrix_norm = self.inner_update_zero(loss_diff_matrix, nv_tensor_list, grads, U)
            para_update_list.append(para_update)
            H_matrix_norm_list.append(H_matrix_norm)

        grads_trpo = torch.mean(torch.stack(para_update_list), dim=0)
        old_losses = torch.mean(torch.stack(old_losses, dim=0))
        H_matrix_norm_mean = np.mean(H_matrix_norm_list)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes_list,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads_trpo,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier
        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes_list, old_pis=old_pis)
            improve = loss - old_losses
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())
        return episodes_list, H_matrix_norm_mean

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                    - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0,
                    weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                    weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def surrogate_loss_gaus(self, episodes, old_pi=None):
        (train_episodes, valid_episodes) = episodes
        params = self.adapt(train_episodes)
        with torch.set_grad_enabled(old_pi is None):
            pi = self.policy(valid_episodes.observations, params=params)
           
            if old_pi is None:
                old_pi = detach_distribution(pi)

            values = self.baseline(valid_episodes)
            advantages = valid_episodes.gae(values, tau=self.tau)
            advantages = weighted_normalize(advantages,
                weights=valid_episodes.mask)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                - old_pi.log_prob(valid_episodes.actions))
            if log_ratio.dim() > 2:
                log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)

            loss = -weighted_mean(ratio * advantages, dim=0,
                weights=valid_episodes.mask)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                weights=mask)
            pi = detach_distribution(pi)

        return (loss, kl, pi)

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
