import numpy as np
from tqdm import tqdm
from opelab.core.baseline import Baseline
from opelab.core.data import DataType
from opelab.core.policy import Policy
import random
import jax 
import torch

class OnPolicy(Baseline):

    def evaluate(self, data:DataType, target:Policy=None, behavior:Policy=None, gamma:float=1.0, reward_estimator=None) -> float:
        total_reward = 0.0
        normalizer = 0.0
        for tau in data:
            discounted_t = 1.0
            for reward in tau['rewards']:
                total_reward += reward * discounted_t
                normalizer += discounted_t
                discounted_t *= gamma
        return total_reward / len(data)
        

class IS(Baseline):
    def evaluate(self, data: DataType, target: Policy, behavior: Policy,
                 gamma: float = 1.0, reward_estimator=None) -> float:
        data = random.choices(data, k=1024)
        N = len(data)

        max_T = max(len(tau['rewards']) for tau in data)
        if gamma == 1.0:
            gamma_pows_full = None  # not needed
        else:
            gamma_pows_full = np.power(gamma, np.arange(max_T, dtype=np.float64))

        mean_est_reward = 0.0

        for tau in tqdm(data):
            s, a, r = tau['states'], tau['actions'], tau['rewards']

            try:
                tp = target.vectorized_prob(s, a)
                bp = behavior.vectorized_prob(s, a)

                # convert to 1D numpy arrays (supports torch or numpy)
                tp_np = tp.detach().cpu().numpy().reshape(-1) if hasattr(tp, "detach") else np.asarray(tp).reshape(-1)
                bp_np = bp.detach().cpu().numpy().reshape(-1) if hasattr(bp, "detach") else np.asarray(bp).reshape(-1)
            except AttributeError:
                # if vectorized_prob is unavailable, use original scalar loop 
                tp_np = np.array([target.prob(si, ai) for si, ai in zip(s, a)], dtype=np.float64)
                bp_np = np.array([behavior.prob(si, ai) for si, ai in zip(s, a)], dtype=np.float64)

            # sum of log importance ratios over the trajectory
            log_tau_ratio = np.log(tp_np).sum() - np.log(bp_np).sum()

            # discounted return of this trajectory (same as your loop)
            r_np = np.asarray(r, dtype=np.float64).reshape(-1)
            T = r_np.shape[0]
            if gamma == 1.0:
                total_reward = r_np.sum()
            else:
                total_reward = np.dot(r_np, gamma_pows_full[:T])

            # accumulate IS estimate
            mean_est_reward += total_reward * np.exp(log_tau_ratio)

        mean_est_reward /= float(N)
        print(mean_est_reward)
        return float(mean_est_reward)
                

class ISStepwise(Baseline):
    def evaluate(self, data: DataType, target: Policy, behavior: Policy,
                 gamma: float = 1.0, reward_estimator=None) -> float:
        epsilon = 1e-12

        data = random.choices(data, k=1024)
        N = len(data)
        max_traj_length = max(len(tau['states']) for tau in data)

        log_prob_ratios = np.zeros((N, max_traj_length))
        rewards = np.zeros((N, max_traj_length))
        mask = np.zeros((N, max_traj_length))

        for i in tqdm(range(N)):
            tau = data[i]
            s, a, r = tau['states'], tau['actions'], tau['rewards']
            T = s.shape[0]

            tp = target.vectorized_prob(s, a).clamp(min=epsilon)
            bp = behavior.vectorized_prob(s, a).clamp(min=epsilon)

            lpr = (tp.log() - bp.log()).reshape(-1).cpu().numpy()

            r_np = np.asarray(r).reshape(-1)

            rewards[i, :T] = r_np[:T]
            log_prob_ratios[i, :T] = np.clip(lpr, -4.0, 2.0)
            mask[i, :T] = 1.0

        # cumulative log-weights and unnormalized weights
        cumulative_lpr = np.cumsum(log_prob_ratios, axis=1)
        cum_probs = np.exp(cumulative_lpr)  

        discount = (gamma ** np.arange(max_traj_length))[None, :]

        weighted_per_step_sum = np.sum(rewards * discount * cum_probs * mask, axis=0)
        avg_step = weighted_per_step_sum / N
        estimated_return = float(np.sum(avg_step))
        return estimated_return


class WeightedIS(Baseline):
    
    def evaluate(self, data:DataType, target:Policy, behavior:Policy, gamma:float=1.0, reward_estimator=None) -> float:
        total_rho = 0.
        epsilon = 1e-6
        mean_est_reward = 0.0
        
        data = random.choices(data, k=1024)
        max_traj_length = max([len(tau['states']) for tau in data])
        
        log_prob_ratios = np.zeros((len(data), max_traj_length))
        rewards = np.zeros((len(data), max_traj_length))
        mask = np.zeros((len(data), max_traj_length))
        
        for i in tqdm(range(len(data))):
            tau = data[i]
            s = tau['states']
            a = tau['actions']
            r = tau['rewards']
            traj_length = s.shape[0]
            
            target_pr = target.prob(s,a)
            behaviour_pr = behavior.vectorized_prob(s,a)
            
            rewards[i, :traj_length] = r 
            
            log_prob_ratio_at_step = np.log(target_pr.detach().cpu()) - np.log(behaviour_pr.detach().cpu())
            
            log_prob_ratios[i, :traj_length] = np.clip(log_prob_ratio_at_step, -6., 2.)
            mask[i, :traj_length] = 1
        
        cumulative_lpr = np.sum(log_prob_ratios, axis=-1)
        
        discount_weights = gamma ** np.arange(max_traj_length)[None,:]
        
        weighed_sum_of_rewards = np.exp(cumulative_lpr) * np.sum(discount_weights * rewards, axis=-1)
        numerator = np.sum(weighed_sum_of_rewards)
        
        denominator = np.sum(np.exp(cumulative_lpr))+1e-10        
        print(numerator/denominator)
        return numerator/denominator
"""        weighted_reward_sum_at_step = np.sum(rewards * discount_weights * cum_probs * mask, axis=0)
        sum_of_weights = np.sum(cum_probs*mask, axis=0) + 1e-10
        
        avg_step = weighted_reward_sum_at_step / sum_of_weights
        
        estimated_return  = np.sum(avg_step)
        print(estimated_return)
        
        
        data = random.choices(data, k=200)
        numerator = 0
        denominator = 0
        for tau in tqdm(data):
            total_reward = 0.0
            log_tau_ratio = 0.0
            discounted_t = 1.0
            normalizer = 0.0
            for state, action, reward in zip(tau['states'], tau['actions'], tau['rewards']):
                target_prob = target.prob(state, action)
                behaviour_prob = behavior.prob(state, action)

                #print(f'P(Target) = {target_prob}, P(behaviour) = {behaviour_prob}')

                log_tau_ratio += np.log(max(target_prob, epsilon)) - np.log(max(behaviour_prob, epsilon))
                
                #print('log tau', log_tau_ratio)
                
                total_reward += reward * discounted_t
                normalizer += discounted_t
                discounted_t *= gamma
            
            numerator += total_reward * max(np.exp(log_tau_ratio),epsilon)
            denominator += max(np.exp(log_tau_ratio), epsilon)
            avr_reward = total_reward #/ normalizer
            tau_ratio = log_tau_ratio
            total_rho += tau_ratio
            #print(tau_ratio)
            
            #print('tau_ratio', tau_ratio) 
            
            mean_est_reward += np.exp(tau_ratio) * avr_reward
            
        avr_rho = jax.tree_util.tree_reduce(jax.numpy.add, jax.tree_util.tree_map(np.exp, total_rho))
        #print(avr_rho)
        print('Return:', numerator/denominator, numerator, denominator)
        return numerator / denominator 
"""    

class WeightedISStepwise(Baseline):
    
    def evaluate(self, data:DataType, target:Policy, behavior:Policy, gamma:float=1.0, reward_estimator=None) -> float:
        log_policy_ratios, REW = [], []
        len_each = None
        epsilon = 1e-20

        data = random.choices(data, k=50)

        #rewards = 0
        #weights = 0
        numerator = 0
        denominator = 0

        #iterate over trajectories
        for tau in tqdm(data):
            #initialize storage for log-policy and rewards
            log_policy_ratio, rew = [], []
            discounted_t = 1.0
            normalizer = 0.0
            
            #iterate over time-steps
            for state, action, reward in zip(tau['states'], tau['actions'], tau['rewards']):
                #collect target probability
                target_prob = target.prob(state, action)
                
                #collect behaviour probability
                behaviour_prob = behavior.prob(state, action)
                
                #calculate log probability ratio
                log_pr = np.log(target_prob + epsilon) - np.log(behaviour_prob + epsilon)
                
                #if log prob already exists, roll it forward to this step, otherwise simply set it to log_pr
                if log_policy_ratio:
                    log_policy_ratio.append((log_pr + log_policy_ratio[-1]))
                else:
                    log_policy_ratio.append(log_pr)
                    
                #add reward
                rew.append(reward)
                
                #calculate normalizer
                normalizer += discounted_t
                
                #update discount factor
                discounted_t *= gamma
            
            
            #update list of log prob trajectories
            log_policy_ratios.append(log_policy_ratio)
            
            #update list of rewards
            REW.append(rew)
            
            
        def is_list_leaf(x):
            return isinstance(x[0], float)

        def make_arange_tree(pytree):
            """
            Creates a new tree with the same structure as the input pytree,
            where each list leaf is replaced with jnp.arange(len(list))
            """
            def leaf_to_arange(leaf):
                return jax.numpy.arange(len(leaf)).tolist()
            
            return jax.tree_util.tree_map(
                leaf_to_arange, 
                pytree,
                is_leaf=is_list_leaf
            )
        ar = make_arange_tree(log_policy_ratios)


        #get for each time-step the discount factor times the probability ratio
        rho = jax.tree_util.tree_map(lambda x, y: gamma ** y * np.exp(x) , log_policy_ratios, ar)
        
        
        REW = REW

        #denominator is \sum_{N}\sum_{T} rho_{i,t}
        #numerator is \sum_{N}\sum_{T}rew_{i,t}*rho_{i,t}
        denominator = jax.tree_util.tree_reduce(jax.numpy.add, rho)
        numerator = jax.tree_util.tree_reduce(jax.numpy.add, jax.tree_util.tree_map(jax.numpy.multiply, rho, REW))
        #estimator is numerator / denominator
        return numerator / denominator 
        
class WeightedISStepwiseV2(Baseline):
    def evaluate(self, data: DataType, target: Policy, behavior: Policy,
                 gamma: float = 1.0, reward_estimator=None) -> float:
        log_policy_ratios, REW = [], []
        len_each = None
        epsilon = 1e-15

        data = random.choices(data, k=1024)

        min_den, max_den = np.inf, -1 * np.inf
        max_traj_length = max(len(tau['states']) for tau in data)

        N = len(data)
        log_prob_ratios = np.zeros((N, max_traj_length))
        rewards = np.zeros((N, max_traj_length))
        mask = np.zeros((N, max_traj_length))

        # Compute log prob ratios per trajectory
        for i in tqdm(range(N)):
            tau = data[i]
            s = tau['states']
            a = tau['actions']
            r = tau['rewards']
            traj_length = s.shape[0]

            target_pr_t = torch.clamp(target.vectorized_prob(s, a), min=epsilon)
            behavior_pr_t = torch.clamp(behavior.vectorized_prob(s, a), min=epsilon)

            log_prob_ratio_at_step = (torch.log(target_pr_t) - torch.log(behavior_pr_t)).reshape(-1).cpu().numpy()

            r_np = np.asarray(r).reshape(-1)

            rewards[i, :traj_length] = r_np[:traj_length]
            log_prob_ratios[i, :traj_length] = np.clip(log_prob_ratio_at_step[:traj_length], -6.0, 2.0)
            mask[i, :traj_length] = 1.0

        # Compute cumulative ratios
        cumulative_lpr = np.cumsum(log_prob_ratios, axis=1)

        logw = np.where(mask == 1.0, cumulative_lpr, -np.inf)  # exclude padded positions in log-space

        # per-timestep stability shift using only valid entries
        lpr_offset = np.max(logw, axis=0)
        lpr_offset[~np.isfinite(lpr_offset)] = 0.0  # if a whole column is invalid (shouldn't happen)

        # exponentiate only masked, shifted entries
        cum_probs = np.exp(logw - lpr_offset[None, :])

        den = np.sum(cum_probs * mask, axis=0) + 1e-10  # sum of weights per t
        norm_w = (cum_probs * mask) / den[None, :]      # per-t normalized weights

        discount = (gamma ** np.arange(max_traj_length))[None, :]

        # per-timestep SNIS expectation; sum over i (not mean), then sum over t
        snis_per_t = np.sum(discount * rewards * norm_w, axis=0)
        estimated_return = float(np.sum(snis_per_t))
        return estimated_return
    """
        max_traj_length = max([len(tau['states']) for tau in data])
        
        log_prob_ratios = np.zeros((len(data), max_traj_length))
        rewards = np.zeros((len(data), max_traj_length))
        mask = np.zeros((len(data), max_traj_length))
        
        for i in tqdm(range(len(data))):
            tau = data[i]
            s = tau['states']
            a = tau['actions']
            r = tau['rewards']
            traj_length = s.shape[0]
            
            target_pr = target.prob(s,a)
            behaviour_pr = behavior.vectorized_prob(s,a)
            
            rewards[i, :traj_length] = r 
            
            log_prob_ratio_at_step = np.log(target_pr.detach().cpu()) - np.log(behaviour_pr.detach().cpu())
            
            log_prob_ratios[i, :traj_length] = np.clip(log_prob_ratio_at_step, -6., 2.)
            mask[i, :traj_length] = 1
            
        cumulative_lpr = np.cumsum(log_prob_ratios, axis=1)
        lpr_offset = cumulative_lpr.max(axis=0)
        cum_probs = np.exp(cumulative_lpr - lpr_offset)
        
        discount_weights = mask* gamma ** np.arange(max_traj_length)[None,:]
        
        
        
        weighted_reward_sum_at_step = np.sum(rewards * discount_weights * cum_probs * mask, axis=0)
        sum_of_weights = np.sum(cum_probs*mask, axis=0) + 1e-10
        
        avg_step = weighted_reward_sum_at_step / sum_of_weights
        
        estimated_return  = np.sum(avg_step)
        print(estimated_return)
        #estimator is numerator / denominator
        return estimated_return
        
    """