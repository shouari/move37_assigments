import gym
import numpy as np
import time

""" Evaluates policy by using it to run 
    policy: the policy to be used.
    gamma:Out discount factor.
    total reward: real value of the total reward recieved by agent under policy.
    """
def run_episode(env,policy, gamma, render= False):
	obs = env.reset()
	totalreward = 0
	step_idx =0
	while True:
		if render:
			env.render()

		obs, reward, done, _ =env.step(int(policy[obs]))
		totalreward += (gamma**step_idx*reward)
		step_idx +=1

		if done:
			break
	return totalreward

def evaluate_policy(env, policy, gamma = 1.0, n =100):
	scores = [run_episode(env, policy, gamma=gamma, render=False) for _ in range(n) ]
	return np.mean(scores)
# Evaluates a policy by running it n times.
    #returns:
   # average total reward

	

def extract_policy(v, gamma= 1.0):
	#Extract policy given a Value function
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			for next_sr in env.P[s][a]:
				q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
		policy[s]=np.argmax(q_sa)
	return policy

def compute_policy_v(env, policy, gamma = 1.0):

	v = np.zeros(env.nS) #initiate value-function	
	eps = 1e-10

	while True:
		prev_v = np.copy(v)

		for s in range(env.nS):
			policy_a = policy[s]
			v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
		if (np.sum(np.fabs(prev_v - v) )<= eps):
			#converged
			break
	return v

def policy_iteration(env, gamma):
	policy = np.random.choice(env.nA, size=(env.nS))
	max_iterations = 200000
	

	for i in range(max_iterations):
		old_policy_v = compute_policy_v(env, policy, gamma)
		new_policy = extract_policy(old_policy_v, gamma)
		if (np.all(policy ==new_policy)):
			print("policy converged at step %d." %(i+1))
			break
		policy =new_policy
	return policy
	
if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    gamma = 1.0
    start_time = time.time()
    env = gym.make(env_name)
    optimal_policy = policy_iteration(env, gamma)
    scores = evaluate_policy(env, optimal_policy, gamma)
    end_time = time.time()
    print("Time taken %4.4f seconds." %(end_time-start_time))
    print('average score = ', np.mean(scores))
