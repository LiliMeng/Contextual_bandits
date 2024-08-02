# Contextual_bandits

## Empirical and theoretical evidence on high-quality feedback lead to worse performance

Empirical and theoretical evidence in the literature suggests that high-quality feedback can sometimes lead agents in contextual bandits to focus too narrowly on perceived optimal actions. This narrow focus arises from the strong signal provided by accurate feedback, which can make the agent overly confident in certain actions, potentially leading to a lack of exploration. Here's a detailed explanation, including a simulation to illustrate this effect and a plot showing how the distribution of actions evolves over iterations.

### Theoretical Evidence

1. **Exploitation Over Exploration**:
   - When feedback is consistently accurate and provides clear guidance on the best actions, agents may prioritize exploitation to maximize immediate rewards. This behavior aligns with the exploration-exploitation trade-off in reinforcement learning, where the presence of a strong feedback signal can skew the balance toward exploitation.

2. **Overconfidence in Feedback**:
   - High-quality feedback reduces uncertainty, leading the agent to be more confident in its action choices. The agent might under-explore because it believes the feedback's indicated actions are close to optimal, which can result in missing alternative actions that could be better in different contexts or in the future.

3. **Feedback-Induced Bias**:
   - Even accurate feedback can be biased toward certain actions, especially if it doesn't cover the entire action space comprehensively. This bias can reinforce specific actions repeatedly, reducing the diversity of actions explored.

### Empirical Evidence

Empirical studies have shown that:
- Agents with access to reliable human feedback often exhibit lower exploration rates.
- Exploration strategies like epsilon-greedy can help maintain exploration, but without deliberate exploration strategies, high-quality feedback can inadvertently limit exploration.

### Simulation and Plotting Action Distribution

To illustrate this concept, let's simulate a simple contextual bandit problem where we compare the action distribution with and without high-quality feedback.

#### Simulation Code

```python
import numpy as np
import matplotlib.pyplot as plt

class ContextualBandit:
    def __init__(self, n_contexts, n_actions):
        self.n_contexts = n_contexts
        self.n_actions = n_actions
        # Random reward probabilities for each context-action pair
        self.probabilities = np.random.rand(n_contexts, n_actions)

    def get_reward(self, context, action):
        # Generate reward based on the probability for the context-action pair
        return 1 if np.random.rand() < self.probabilities[context, action] else 0

class EpsilonGreedyAgent:
    def __init__(self, n_contexts, n_actions, epsilon=0.1, feedback_quality=1.0):
        self.epsilon = epsilon
        self.n_contexts = n_contexts
        self.n_actions = n_actions
        self.feedback_quality = feedback_quality
        self.Q = np.zeros((n_contexts, n_actions))
        self.action_counts = np.zeros((n_contexts, n_actions))
    
    def select_action(self, context):
        if np.random.rand() < self.epsilon * self.feedback_quality:
            return np.random.choice(self.n_actions)  # Explore
        else:
            return np.argmax(self.Q[context])  # Exploit
    
    def update(self, context, action, reward):
        self.action_counts[context, action] += 1
        self.Q[context, action] += (reward - self.Q[context, action]) / self.action_counts[context, action]

# Simulation parameters
n_contexts = 3
n_actions = 3
n_iterations = 1000
feedback_quality_high = 0.1  # Lower epsilon due to confidence
feedback_quality_low = 0.5   # Higher epsilon due to uncertainty

# Create bandit environment
bandit = ContextualBandit(n_contexts, n_actions)

# Create agents with different feedback qualities
agent_high_feedback = EpsilonGreedyAgent(n_contexts, n_actions, epsilon=0.1, feedback_quality=feedback_quality_high)
agent_low_feedback = EpsilonGreedyAgent(n_contexts, n_actions, epsilon=0.1, feedback_quality=feedback_quality_low)

# Track action distributions
action_distribution_high = np.zeros((n_iterations, n_actions))
action_distribution_low = np.zeros((n_iterations, n_actions))

# Simulate learning process
for t in range(n_iterations):
    context = np.random.choice(n_contexts)
    
    # High-quality feedback
    action_high = agent_high_feedback.select_action(context)
    reward_high = bandit.get_reward(context, action_high)
    agent_high_feedback.update(context, action_high, reward_high)
    action_distribution_high[t, :] = agent_high_feedback.Q[context] / agent_high_feedback.Q[context].sum()
    
    # Low-quality feedback
    action_low = agent_low_feedback.select_action(context)
    reward_low = bandit.get_reward(context, action_low)
    agent_low_feedback.update(context, action_low, reward_low)
    action_distribution_low[t, :] = agent_low_feedback.Q[context] / agent_low_feedback.Q[context].sum()

# Plot the action distribution over iterations for high and low-quality feedback
fig, axes = plt.subplots(n_contexts, 2, figsize=(12, 8), sharey=True)

for c in range(n_contexts):
    axes[c, 0].plot(action_distribution_high[:, :], label=[f'Action {a+1}' for a in range(n_actions)])
    axes[c, 0].set_title(f'High-Quality Feedback - Context {c+1}')
    axes[c, 0].set_xlabel('Iterations')
    axes[c, 0].set_ylabel('Probability')
    axes[c, 0].legend()
    axes[c, 0].grid(True)

    axes[c, 1].plot(action_distribution_low[:, :], label=[f'Action {a+1}' for a in range(n_actions)])
    axes[c, 1].set_title(f'Low-Quality Feedback - Context {c+1}')
    axes[c, 1].set_xlabel('Iterations')
    axes[c, 1].legend()
    axes[c, 1].grid(True)

plt.tight_layout()
plt.show()
```

### Explanation

- **High-Quality Feedback**: The agent uses a lower exploration rate (\(\epsilon\)) because it is confident in the feedback's guidance. The action distribution quickly converges to favor actions suggested by the feedback.
  
- **Low-Quality Feedback**: The agent explores more widely due to higher uncertainty (\(\epsilon\) is effectively larger). This results in a broader action distribution, allowing the agent to explore alternative actions more frequently.

### Conclusion

The simulation shows that high-quality feedback can lead to a narrower focus on actions deemed optimal by the feedback, resulting in reduced exploration. In contrast, low-quality feedback encourages broader exploration, helping the agent discover other potentially rewarding actions. Balancing exploration and exploitation, even with high-quality feedback, is crucial for maintaining adaptability and optimizing long-term performance.


Certainly! Let’s break down the plots generated by the simulation code to understand how high-quality and low-quality feedback affect action distribution in contextual bandits:

### Simulation Overview

The simulation involves a contextual bandit problem with three contexts and three actions. Two agents are used: one receiving high-quality feedback and another receiving low-quality feedback. The difference in feedback quality is modeled through the exploration parameter \(\epsilon\) in an epsilon-greedy strategy.

### Plot Explanation

Each subplot in the figures represents the action distribution over iterations for a specific context. The x-axis represents the number of iterations (or time steps), and the y-axis represents the probability of selecting each action.

#### High-Quality Feedback

1. **Convergence to Optimal Actions**:
   - In each context, the agent with high-quality feedback quickly converges to a narrow distribution, favoring one or a few actions. This is because the feedback effectively informs the agent about which actions are most likely to yield rewards.
   - The plot shows steep increases in the probability of selecting certain actions, indicating rapid exploitation of the feedback.

2. **Reduced Exploration**:
   - The lower exploration rate (\(\epsilon\)) means the agent is less likely to explore alternative actions. As a result, the action distribution is sharply focused on the perceived optimal actions, as guided by the feedback.
   - This can be seen in how quickly the distribution stabilizes, with certain actions being consistently favored over others.

#### Low-Quality Feedback

1. **Broader Action Distribution**:
   - In the case of low-quality feedback, the agent maintains a more diverse action distribution. This is due to the higher exploration rate, allowing the agent to try different actions more frequently.
   - The plot shows a more gradual change in action probabilities, reflecting ongoing exploration and less certainty about which actions are best.

2. **Sustained Exploration**:
   - The agent continues to explore alternative actions, which can be beneficial in discovering actions that might not have been favored by the feedback but are potentially rewarding.
   - This is represented by a less stable action distribution, with probabilities fluctuating as the agent gathers more information.

### Key Observations

- **High-Quality Feedback**:
  - Leads to faster convergence to specific actions, but may result in less exploration of the action space.
  - The agent relies heavily on feedback, which can lead to exploitation of known actions at the expense of discovering new opportunities.

- **Low-Quality Feedback**:
  - Encourages more exploration, leading to a more balanced distribution across actions.
  - The agent is more likely to discover alternative actions that might be rewarding but were not initially favored by the feedback.

### Implications

- **Adaptability**: High-quality feedback may limit adaptability by narrowing the focus too quickly on specific actions. In contrast, low-quality feedback maintains broader exploration, which can be beneficial in dynamic environments where optimal actions might change.

- **Exploration-Exploitation Trade-off**: Balancing exploration and exploitation is crucial. High-quality feedback should be used to guide but not completely dictate the agent's actions, allowing for ongoing exploration to adapt to changes and discover new strategies.

These plots and their interpretations demonstrate the importance of managing feedback and exploration strategies to optimize learning and performance in contextual bandits.
