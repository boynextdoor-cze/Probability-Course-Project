import numpy as np
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
import math

N = 6000
EPS = 200
C = [0.1*k for k in range(21)]
prob = [0.8, 0.6, 0.5]
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

# The average reward at each time slot for different epsilons
agg_avg_reward = np.zeros((21, N))

for times in range(EPS):
    theta = np.zeros((21, 3))
    count = np.zeros((21, 3))
    for i in range(len(C)):
        init_R = []
        cum_reward = [0]
        # Initialization when t=1,2,3:
        for t in range(3):
            p = prob[t]
            reward = random.binomial(1, p)
            init_R.append(reward)
            cum_reward.append((reward+t*cum_reward[-1])/(t+1))
        theta[i] = np.array(init_R)
        count[i] = np.ones(3)
        c = C[i]
        for t in range(3, N):
            # Compute the comprehensive value of exploration and exploitation to determine which arm to choose:
            EE_mix = np.array(theta[i]+c*np.sqrt(2*math.log(t+1)/count[i]))
            arm = np.argmax(EE_mix)
            # Compute cumulative average reward up to time slot t:
            reward = random.binomial(1, prob[arm])
            cum_reward.append((reward+t*cum_reward[-1])/(t+1))
            # Update the percent conversion and count of arm I:
            count[i][arm] += 1
            theta[i][arm] += (reward-theta[i][arm])/count[i][arm]
        # Summarize the cumulative average reward in a certain experiment:
        agg_avg_reward[i] += np.array(cum_reward[1:])
# Compute the average cumulative average reward over 200 experiments:
agg_avg_reward /= np.float(EPS)

# Visualization:
x = np.array(C)
y = agg_avg_reward[:, N-1]
plt.xlabel("Confidence value")
plt.ylabel("Average Aggregate Reward")
plt.title("Average Aggregate Reward given C")
plt.ylim(0.7, 0.85)
plt.plot(x, y, '--')
plt.plot(x, y, 'o')
plt.show()