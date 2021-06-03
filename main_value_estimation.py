import random
import torch
import gym
import argparse
import os
import time
import numpy as np
# ------------------------------
from DDPG import DDPG_value_estimation
# -------------------------------
from TD3 import TD3_value_estimation
# -------------------------------
from utils import replay_buffer

from spinupUtils.logx import EpochLogger
from spinupUtils.run_utils import setup_logger_kwargs

def discount_cumsum(x, discount):
	"""
	magic from rllab for computing discounted cumulative sums of vectors.
	i.e.
	input: 
		vector x: [x0, x1, x2]
	output:
		[x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
	"""
	import scipy.signal
	return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def test_agent(policy, eval_env, logger, eval_episodes=10):
	for _ in range(eval_episodes):
		state, done, ep_ret, ep_len = eval_env.reset(), False, 0, 0
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			ep_ret += reward
			ep_len += 1
		logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


def value_estimate(replay_buffer, policy, eval_env, logger, batch_size=1000, max_episode_steps=500):
	batch_state, mjsim_state_list = replay_buffer.sample_sim_state(batch_size)
	V_value = policy.predict_V(batch_state)
	total_Gt = 0
	logger.store(EstimateValue=V_value)
	reward_buf = np.zeros(env._max_episode_steps+1, dtype=np.float32)
	for mjsim_state in mjsim_state_list:
		eval_env.reset()
		eval_env.sim.set_state(mjsim_state["mjsim_state"])
		state, done, ep_ret, ep_len, timeout_done = mjsim_state["env_state"], False, 0, 0, False
		while not (done or timeout_done):
			ep_len += 1
			if args.policy.startswith("SAC"):
				action = policy.select_action(np.array(state), deterministic=True)
			else:
				action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			timeout_done = (ep_len == max_episode_steps)
			reward_buf[ep_len-1] = reward
			ep_ret += reward

		if timeout_done:
			_, v = policy.select_action(np.array(state), return_critic_value=True)
		else:
			v = 0
		ep_len += 1
		reward_buf[ep_len-1] = v
		
		path_slice = slice(0, ep_len)
		reward_array = reward_buf[path_slice]
		discounted_reward = discount_cumsum(reward_array, policy.discount)[0]
		total_Gt += discounted_reward 
		logger.store(
			EstimateDiscountedEpRet=discounted_reward, EstimateEpRet=ep_ret, EstimateEpLen=ep_len-1
		)
	Gt = total_Gt / batch_size
	bias = V_value - Gt
	logger.store(EstimateGt=Gt)
	logger.store(EstimateBias=bias)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3_value_estimation")  # Policy name
	parser.add_argument("--env", default="HalfCheetah-v2")           # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=3e6, type=int)    # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                  # Discount factor
	parser.add_argument("--tau", default=0.005)                      # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
	parser.add_argument("--alpha", default=0.2, type=float)          # For sac entropy
	parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--exp_name", type=str)       				 # Name for algorithms
	parser.add_argument("--estimate_times", default=30, type=int)    # times for doing value estimation
	
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print(f"---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print(f"---------------------------------------")

	args.estimate_freq = int(args.max_timesteps / args.estimate_times)

	# Make envs
	env = gym.make(args.env)
	eval_env = gym.make(args.env)
	estimation_env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	eval_env.seed(args.seed)  # eval env for evaluating the agent
	estimation_env.seed(args.seed)  # estimation env for rolling out states sampled from replay buffer
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	# ----------------------------------------------
	if args.policy == "DDPG_value_estimation":
		# if the formal argument defined in function `DDPG()` are regular params, can pass `**-styled` actual argument.
		policy = DDPG_value_estimation.DDPG_ValueEstimate(**kwargs)
	elif args.policy == "TD3_value_estimation":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3_value_estimation.TD3_ValueEstimate(**kwargs)
	else:
		raise ValueError(f"Invalid Policy: {args.policy}!")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		if not os.path.exists(f"./models/{policy_file}"):
			assert f"The loading model path of `../models/{policy_file}` does not exist! "
		policy.load(f"./models/{policy_file}")

	# Setup loggers
	logger_kwargs = setup_logger_kwargs(f"{args.exp_name}-reward", args.seed, datestamp=False)
	logger_kwargs1 = setup_logger_kwargs(f"{args.exp_name}-estimation", args.seed, datestamp=False)
	logger = EpochLogger(**logger_kwargs)
	logger1 = EpochLogger(**logger_kwargs1)

	_replay_buffer = replay_buffer.ValueEstimateReplayBuffer(state_dim, action_dim)
	
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	start_time = time.time()

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < int(args.start_timesteps):
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)

		# If env stops when reaching max-timesteps, then `done_bool = False`, else `done_bool = True`
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0  

		# Store data in replay buffer
		save_state = env.sim.get_state()
		_replay_buffer.add(state, action, next_state, reward, done_bool, save_state)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= int(args.start_timesteps):
			policy.train(_replay_buffer, args.batch_size)

		if done: 
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			logger.store(EpRet=episode_reward, EpLen=episode_timesteps)
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		if (t + 1) % args.eval_freq == 0:
			test_agent(policy, eval_env, logger)
			if args.save_model: 
				policy.save(f"./models/{file_name}")
			logger.log_tabular("EpRet", with_min_and_max=True)
			logger.log_tabular("TestEpRet", with_min_and_max=True)
			logger.log_tabular("EpLen", average_only=True)
			logger.log_tabular("TestEpLen", average_only=True)
			logger.log_tabular("TotalEnvInteracts", t+1)
			logger.log_tabular("Time", time.time()-start_time)
			logger.dump_tabular()
		
		if (t + 1) % args.estimate_freq == 0:
			value_estimate(_replay_buffer, policy, estimation_env, logger1)
			logger1.log_tabular("EstimateValue", average_only=True)
			logger1.log_tabular("EstimateDiscountedEpRet", with_min_and_max=True)
			logger1.log_tabular("EstimateEpRet", with_min_and_max=True)
			logger1.log_tabular("EstimateEpLen", average_only=True)
			logger1.log_tabular("EstimateBias", average_only=True)
			logger1.log_tabular("EstimateGt", average_only=True)
			logger1.log_tabular("TotalEnvInteracts", t+1)
			logger1.log_tabular("Time", time.time()-start_time)
			logger1.dump_tabular()

		