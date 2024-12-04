import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy, Policy
from student_submissions.s2250013.policy2250013 import Policy2250013

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

def simulate_with(policy_cls: type[Policy]):
    if not issubclass(policy_cls, Policy):
        raise ValueError('Error')

    policy = policy_cls()
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)
    ep = 0
    while ep < NUM_EPISODES:
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

def student_policy():
    # Uncomment the following code to test your policy
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)

    policy2250013 = Policy2250013()
    for _ in range(200):
        action = policy2250013.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)

        if terminated or truncated:
            observation, info = env.reset()

if __name__ == "__main__":
    # Test GreedyPolicy
    # simulate_with(GreedyPolicy)

    # Test RandomPolicy
    # simulate_with(RandomPolicy)

    student_policy()


env.close()