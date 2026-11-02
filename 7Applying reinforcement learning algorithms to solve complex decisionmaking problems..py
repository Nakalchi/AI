import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

n_states = 20
n_actions = env.action_space.n
episodes = 500

learning_rate = 0.1
discount_factor = 0.99

epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

state_bins = [
    np.linspace(-4.8, 4.8, n_states - 1),
    np.linspace(-4.0, 4.0, n_states - 1),
    np.linspace(-0.418, 0.418, n_states - 1),
    np.linspace(-4.0, 4.0, n_states - 1)
]

def discretize_state(state):
    return tuple(
        np.digitize(state[i], state_bins[i])
        for i in range(len(state))
    )

q_table = np.zeros((n_states,) * 4 + (n_actions,))

rewards = []

for episode in range(episodes):

    state_raw, _ = env.reset()
    state = discretize_state(state_raw)

    total_reward = 0
    done = False

    while not done:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state_raw, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_state = discretize_state(next_state_raw)

        q_table[state][action] += learning_rate * (
            reward
            + discount_factor * np.max(q_table[next_state])
            - q_table[state][action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    rewards.append(total_reward)

    print(
        f"Episode {episode + 1}/{episodes} | "
        f"Reward: {total_reward:.0f} | "
        f"Epsilon: {epsilon:.3f}"
    )

plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title("Total Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

np.save("q_table.npy", q_table)
print("Q-table saved as 'q_table.npy'")

test_env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

state_raw, _ = test_env.reset()
state = discretize_state(state_raw)

done = False
total_reward = 0

while not done:

    action = np.argmax(q_table[state])

    next_state_raw, reward, terminated, truncated, _ = test_env.step(action)

    done = terminated or truncated

    state = discretize_state(next_state_raw)

    total_reward += reward

test_env.close()

print(f"\nTotal reward during test: {total_reward}")
