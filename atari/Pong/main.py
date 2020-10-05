"author: Phil Tabor"
import gym
import numpy as np
from dqn import *
from utils import plot_learning_curve, make_env
from gym import wrappers




if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 250

 

    agent = DQN(states=env.observation_space, actions=env.action_space.n,alpha=0.0001,gamma=0.99,epsilon=1,
                epsilon_min=0.1,epsilon_decay=1e-5,replay_buffer_sz=1000,batch=32,path = 'atari_model_eval.txt',
                path_pred = 'atari_model_target.txt')


    if load_checkpoint:
        agent.load_models()

    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    count = 0
    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.e_greedy_policy(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            count += 1
            if not load_checkpoint:
                agent.store(observation, action,
                                     reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)



