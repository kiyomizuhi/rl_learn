from env_config import *
from environment import *
from agent import *

def main(num_episodes=10):
    env = Environment(grid)
    agent = Agent(env)
    datas = []
    for _ in range(num_episodes):
        env.reset() # reset the agent's state to the initial state
        done = False
        s1 = env.states[0].clone()
        data = []
        while not done:
            a = agent.policy(s1)
            s2, r, done = env.step(a)
            s1 = s2
            data.append((s1, s2, a, r, done))
            print(s1, s2, a, r, done)
        datas.append(data)
    return datas

if __name__ == "__main__":
    main()