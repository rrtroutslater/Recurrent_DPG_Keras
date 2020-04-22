from __future__ import print_function
from ActorRDPG import *
from CriticRDPG import *
from EncoderNet import *
import matplotlib.pyplot as plt
import os
import pickle
import math


class AgentRDPG():
    def __init__(self,
                 session,
                 actor,
                 critic,
                 lstm_horizon=4,
                 gamma=0.95,
                 tau_actor=0.1,
                 tau_critic=0.1,
                 ):

        self.sess = session
        self.actor = actor
        self.critic = critic
        self.lstm_horizon = lstm_horizon
        self.gamma = gamma
        self.tau_actor = tau_actor
        self.tau_critic = tau_critic

        self.learning_rate_actor = self.actor.learning_rate
        self.learning_rate_critic = self.critic.learning_rate

        return

    def train_rdpg(self,
                   dataset,
                   num_episode_per_dataset=20,
                   num_update=1,
                   lstm_horizon=10,
                   ):
        """
        """

        loss = []
        for i in range(0, num_episode_per_dataset):

            data_pre_episode, episode = self.extract_episode(
                dataset,
                lstm_horizon=lstm_horizon)

            # extract observations (features) from images in dataset
            obs_pre_episode = data_pre_episode['img_0']
            obs_target_pre_episode = data_pre_episode['img_1']
            obs = episode['img_0']
            obs_target = episode['img_1']

            # forward propagate through all pre-episode data to initialize hidden state
            # 1) get pre-episode actions from actor network, update hidden states of actor, actor target
            # act_pre_episode = self.actor.sample_act(obs_pre_episode)
            act_pre_episode = data_pre_episode['act']
            # is this correct?
            act_target_pre_episode = self.actor.sample_act_target(obs_target_pre_episode)

            # 2) use actions and obs before episode to update Q network hidden states
            q_pre_episode = self.critic.sample_q(obs_pre_episode, act_pre_episode)
            q_target_pre_episode = self.critic.sample_q_target(obs_target_pre_episode, act_target_pre_episode)

            # actions, rewards which occurred during episode
            act = episode['act']
            reward = episode['reward'].T

            # probably can't do more than a single update (to critic?) without re-initializing hidden states
            for i in range(0, num_update):
                # action and Q values from target nets, used for training critic Q function
                act_target = self.actor.sample_act_target(obs_target)
                q_target = self.critic.sample_q_target(obs_target, act_target)

                # regression label
                gamma = 0.95
                reward = np.expand_dims(reward, axis=0)
                y = np.zeros(shape=(1, reward.shape[1], 1))
                if (math.isnan(y[0][0])):
                    print('nan found!')
                    continue
                for i in range(0, reward.shape[1]):
                    y[:,i,:] = reward[:,i,:] + q_target[:,i,:]* gamma ** i
                # y = reward + gamma * q_target

                q_pred = self.critic.sample_q(obs, act, reset_hidden_after_sample=True)
                if (math.isnan(y[0][0])):
                    print('nan found')
                    print('act:\n', act_target[0])
                    print('y target:\n', y)
                    print('q_target:\n', q_target[0].T)
                    print('reward:\n', reward)
                    input('press key to cont:')
                    continue
                
                # get gradients from Q net, which are backpropagated through actor and encoder
                # NOTE: this must be done before the train_on_batch below
                # mu(o)
                act_for_grad = self.actor.sample_act(obs, reset_hidden_after_sample=True)

                # for actor grad update
                dQ_da = self.critic.get_dQ_da_critic(obs, act_for_grad)

                # take a gradient step on the Q network weights
                l = self.critic.train_critic(act, obs, y, num_step=1)
                loss.append(l[0])

                # grad step on actor
                self.actor.apply_gradients(obs, dQ_da, num_step=1)

                # update target networks
                self.actor.update_target_net(self.tau_actor)
                self.critic.update_target_net(self.tau_critic)

        return loss

    def get_act(self,
                obs,
                add_noise=False,
                variance=2):
        """ 
        TODO 
        use heavy-tail distribution instead of gaussian
        """
        act = self.actor.sample_act(obs)[0][0]
        if add_noise:
            act += np.random.randn(self.actor.act_dim) * variance
        return act

    def get_q_pred(self,
                   obs,
                   act,
                   ):
        q = self.critic.sample_q(obs, act)
        return q

    def extract_episode(self,
                        dataset,
                        lstm_horizon):
        """
        """
        # np.random.seed(0)
        n = dataset['img_0'].shape[0]

        good_data = False
        while not good_data:
            idx = np.random.randint(1, n - lstm_horizon - 1)

            data_pre_episode = {}
            data_pre_episode['img_0'] = [dataset['img_0'][:idx]]
            data_pre_episode['img_1'] = [dataset['img_1'][:idx+1]]
            data_pre_episode['act'] = [dataset['act'][:idx, :]]
            data_pre_episode['reward'] = [dataset['reward'][:idx]]

            episode = {}
            episode['img_0'] = [dataset['img_0'][idx:idx+lstm_horizon]]
            episode['img_1'] = [dataset['img_1'][idx+1: idx + lstm_horizon + 1]]
            episode['act'] = np.array([dataset['act'][idx:idx+lstm_horizon, :]])
            episode['reward'] = np.array([dataset['reward'][idx:idx+lstm_horizon]])

            if np.sum(episode['reward']) > 0:# and not math.isnan(episode['reward'][0]):
                good_data = True

            if math.isnan(episode['reward'][0][0]):
                print('found a nan, not using this sample')
                good_data = False
        return data_pre_episode, episode


if __name__ == "__main__":
    # np.random.seed(0)
    data_dir = './training_dicts/'
    fn_list = os.listdir(data_dir)

    # # 8
    # critic = CriticRDPG(session, lstm_horizon=8)  # test_mode=True)
    # actor = ActorRDPG(session, lstm_horizon=8)  # test_mode=True)
    # agent = AgentRDPG(session, actor, critic)
    # loss_8 = []
    # for i in range(3000):
    #     idx = np.random.randint(len(fn_list))
    #     with open(data_dir + fn_list[idx], 'rb') as f:
    #         dataset = pickle.load(f)
    #     l = agent.train_rdpg(dataset, 
    #                         num_episode_per_dataset=1, 
    #                         lstm_horizon=8,
    #                         num_update=1)
    #     for i in range(len(l)):
    #         loss_8.append(l[i])
    # loss_8 = [loss_8[i] / 8 for i in range(0, len(loss_8))]
    # loss_8 = np.convolve(np.array(loss_8), np.ones(40), 'valid') / 40

    # plt.plot(loss_8, label="8")
    # plt.legend(loc="upper right")
    # plt.title('Q-function Loss')
    # plt.xlabel('iter')
    # plt.ylabel('')
    # plt.show()
    # critic.save_model()
    # actor.save_model()

    # # 12
    # critic = CriticRDPG(session, lstm_horizon=12)  # test_mode=True)
    # actor = ActorRDPG(session, lstm_horizon=12)  # test_mode=True)
    # agent = AgentRDPG(session, actor, critic)
    # loss_12 = []
    # for i in range(3000):
    #     idx = np.random.randint(len(fn_list))
    #     with open(data_dir + fn_list[idx], 'rb') as f:
    #         dataset = pickle.load(f)
    #     l = agent.train_rdpg(dataset, 
    #                         num_episode_per_dataset=1, 
    #                         lstm_horizon=12,
    #                         num_update=1)
    #     for i in range(len(l)):
    #         loss_12.append(l[i])
    # loss_12 = [loss_12[i] / 12 for i in range(0, len(loss_12))]
    # loss_12 = np.convolve(np.array(loss_12), np.ones(40), 'valid') / 40

    # plt.plot(loss_12, label="12")
    # plt.legend(loc="upper right")
    # plt.title('Q-function Loss')
    # plt.xlabel('iter')
    # plt.ylabel('')
    # plt.show()
    # critic.save_model()
    # actor.save_model()

    # # 16
    # critic = CriticRDPG(session, lstm_horizon=16)  # test_mode=True)
    # actor = ActorRDPG(session, lstm_horizon=16)  # test_mode=True)
    # agent = AgentRDPG(session, actor, critic)
    # loss_16 = []
    # for i in range(3000):
    #     idx = np.random.randint(len(fn_list))1
    #     with open(data_dir + fn_list[idx], 'rb') as f:
    #         dataset = pickle.load(f)
    #     l = agent.train_rdpg(dataset, 
    #                         num_episode_per_dataset=1, 
    #                         lstm_horizon=16,
    #                         num_update=1)
    #     for i in range(len(l)):
    #         loss_16.append(l[i])
    # loss_16 = [loss_16[i] / 16 for i in range(0, len(loss_16))]
    # loss_16 = np.convolve(np.array(loss_16), np.ones(40), 'valid') / 40

    # plt.plot(loss_16, label="16")
    # plt.legend(loc="upper right")
    # plt.title('Q-function Loss')
    # plt.xlabel('iter')
    # plt.ylabel('')1

    # 20
    learning_rates = [
        0.0000001,
        0.0000005,
        0.000001,
        0.000005,
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
    ]

    tau = 0.1

    for lr in learning_rates:
        session = tf.compat.v1.Session()
        critic = CriticRDPG(
            session, 
            lstm_horizon=20,
            learning_rate=lr,
            )

        actor = ActorRDPG(
            session, 
            lstm_horizon=20,
            learning_rate=lr,
            )
            
        agent = AgentRDPG(
            session, 
            actor, 
            critic,
            tau_actor=tau,
            tau_critic=tau,
            )

        loss_20 = []
        for i in range(0, 8000):
            idx = np.random.randint(len(fn_list))
            # idx = 0
            with open(data_dir + fn_list[idx], 'rb') as f:
                dataset = pickle.load(f)
            l = agent.train_rdpg(dataset, 
                                num_episode_per_dataset=1, 
                                lstm_horizon=20,
                                num_update=1)
            for j in range(len(l)):
                loss_20.append(l[j])
            if i % 200 == 0:
                print('\n----------------------')
                print('iter:\t', i)
                print('loss:\t', loss_20[-1])
                print('learning rate:\n', lr)
        loss_20 = [loss_20[i] / 20 for i in range(0, len(loss_20))]
        loss_20 = np.convolve(np.array(loss_20), np.ones(40), 'valid') / 40

        plt.plot(loss_20, label="20")
        plt.legend(loc="upper right")
        plt.title('Q-function Loss: LR: ' + str(lr) + ' tau: ' + str(tau))
        plt.xlabel('iter')
        plt.ylabel('')
        plt.savefig("q_loss_" + str(lr) + "_" + str(tau) + ".png")
        plt.close()
        critic.save_model(learning_rate=str(lr), tau=str(tau))
        actor.save_model(learning_rate=str(lr), tau=str(tau))
        session.close()

    pass
