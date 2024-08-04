import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import matplotlib.pyplot as plt

class LinearUCBWithEntropyFeedback:
    def __init__(self, n_features, n_actions, alpha=0.5, feedback_strength=0.95):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha  # Reduced exploration
        self.feedback_strength = feedback_strength  # High feedback influence
        
        # Initialize parameters for each action
        self.A = [np.eye(n_features) for _ in range(n_actions)]  # Feature covariance matrix
        self.b = [np.zeros(n_features) for _ in range(n_actions)]  # Reward vector
        
    def select_action(self, context, human_feedback=None):
        # Calculate UCB scores for each action
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            theta = np.linalg.inv(self.A[a]).dot(self.b[a])
            uncertainty = self.alpha * np.sqrt(context.dot(np.linalg.inv(self.A[a])).dot(context))
            p[a] = context.dot(theta) + uncertainty

        # Calculate entropy to decide exploration
        probabilities = softmax(p)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Incorporate human feedback during high uncertainty
        if entropy > 1.0 and human_feedback is not None:
            p = (1 - self.feedback_strength) * p + self.feedback_strength * human_feedback
            probabilities = softmax(p)  # Re-normalize
        
        # Select action with the highest UCB score
        action = np.argmax(probabilities)
        
        return action
    
    def update(self, context, action, reward):
        # Update parameters for the selected action
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

# Load Iris dataset and prepare it for contextual bandits
iris = load_iris()
X = iris.data
y = iris.target
n_samples, n_features = X.shape
n_actions = len(np.unique(y))  # Number of classes

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# High-quality feedback based on a specific feature threshold
def high_quality_feedback(context):
    feedback = np.zeros(n_actions)
    # Suppose the 3rd feature is a strong predictor; adjust based on the dataset specifics
    if context[2] > 0.5:  # More extreme threshold for over-reliance
        feedback[1] = 1.0  # Suggest class 1 (versicolor)
    else:
        feedback[2] = 1.0  # Suggest class 2 (virginica)
    return feedback

# Initialize the agent
agent = LinearUCBWithEntropyFeedback(n_features=n_features, n_actions=n_actions, alpha=0.5, feedback_strength=0.95)

# Track cumulative rewards and accuracy
n_iterations = len(X_train)
cumulative_rewards = np.zeros(n_iterations)
train_accuracy = np.zeros(n_iterations)

for t in range(n_iterations):
    context = X_train[t]
    true_action = y_train[t]

    # Generate high-quality feedback based on the current context
    feedback = high_quality_feedback(context)

    # Agent selects an action based on the context and human feedback
    action = agent.select_action(context, human_feedback=feedback)

    # Receive reward (1 if the action matches the true class, otherwise 0)
    reward = 1 if action == true_action else 0

    # Update the agent's model
    agent.update(context, action, reward)

    # Track cumulative reward
    cumulative_rewards[t] = cumulative_rewards[t - 1] + reward if t > 0 else reward

    # Calculate and track training accuracy
    correct_predictions = np.sum([agent.select_action(X_train[i]) == y_train[i] for i in range(n_iterations)])
    train_accuracy[t] = correct_predictions / (t + 1)

# Evaluate the agent on the test set
correct_predictions = 0
for i in range(len(X_test)):
    context = X_test[i]
    true_action = y_test[i]
    action = agent.select_action(context)
    if action == true_action:
        correct_predictions += 1

test_accuracy = correct_predictions / len(X_test)

# Plot cumulative rewards and training accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cumulative_rewards, label='Cumulative Reward')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward over Time with High-Quality Feedback')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Training vs. Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f'Test Set Accuracy with High-Quality Feature-Based Feedback: {test_accuracy:.2f}')
