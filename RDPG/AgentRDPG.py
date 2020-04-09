from ActorRDPG import *
from CriticRDPG import *
from Encoder import *
from EncoderNet import *
import matplotlib.pyplot as plt
import os
import pickle


class AgentRDPG():
    def __init__(self,
                 session,
                 actor,
                 critic,
                 encoder,
                 lstm_horizon=4,
                 gamma=0.95,
                 tau_actor=0.1,
                 tau_critic=0.1,
                 ):

        self.sess = session
        self.actor = actor
        self.critic = critic
        self.encoder = encoder
        self.lstm_horizon = lstm_horizon
        self.gamma = gamma
        self.tau_actor = tau_actor
        self.tau_critic = tau_critic

        self.learning_rate_actor = self.actor.learning_rate
        self.learning_rate_critic = self.critic.learning_rate

        return

    def train_rdpg(self,
                   dataset,
                   num_episode=20,
                   num_update=1,
                   lstm_horizon=10,
                   ):
        """
        """

        loss = []
        for i in range(0, num_episode):

            data_pre_episode, episode = self.extract_episode(
                dataset,
                lstm_horizon=lstm_horizon)

            # extract observations (features) from images in dataset
            obs_pre_episode = self.encoder.sample_obs(
                data_pre_episode['img_0'])
            obs_target_pre_episode = self.encoder.sample_obs(
                data_pre_episode['img_1'])
            obs = self.encoder.sample_obs(episode['img_0'])
            obs_target = self.encoder.sample_obs(episode['img_1'])

            # forward propagate through all pre-episode data to initialize hidden state
            # 1) get pre-episode actions from actor network, update hidden states of actor, actor target
            # act_pre_episode = self.actor.sample_act(obs_pre_episode)
            act_pre_episode = data_pre_episode['act']
            act_target_pre_episode = self.actor.sample_act_target(
                obs_target_pre_episode)

            # 2) use actions and obs before episode to update Q network hidden states
            q_pre_episode = self.critic.sample_q(
                obs_pre_episode, act_pre_episode)
            q_target_pre_episode = self.critic.sample_q_target(
                obs_target_pre_episode, act_target_pre_episode)

            # get observations, actions, rewards which occurred during episode
            obs = self.encoder.sample_obs(episode['img_0'])
            obs_target = self.encoder.sample_obs(episode['img_1'])
            act = episode['act']
            reward = episode['reward']

            # probably can't do more than a single update without re-initializing hidden states
            for i in range(0, num_update):
                # action and Q values from target nets, used for training critic Q function
                act_target = self.actor.sample_act_target(obs_target)
                q_target = self.critic.sample_q_target(obs_target, act_target)

                # regression label
                gamma = 0.95

                # print('reard shape:', reward.shape)
                # print('reard shape:', np.expand_dims(reward, axis=2).shape)

                # TODO: maybe more explicit calculation of gradients here
                # y = reward + gamma * q_target
                y = np.expand_dims(reward, axis=2) + gamma * q_target

                # get gradients from Q net, which are backpropagated through actor and encoder
                # NOTE: this must be done before the train_on_batch below
                # mu(o)
                act_for_grad = self.actor.sample_act(obs)

                # for actor grad update
                dQ_da = self.critic.get_dQ_da_critic(obs, act_for_grad)

                # for encoder update
                dL_do = self.critic.get_dL_do_critic(y, obs, act)
                dQ_do = self.actor.get_dQ_do_actor(obs, dQ_da)

                # take a gradient step on the Q network weights
                l = self.critic.net.train_on_batch([act, obs], y)
                loss.append(l)

                # grad step on actor
                self.actor.apply_gradients(obs, dQ_da, num_step=1)

                # grad step on encoder
                self.encoder.apply_gradients_to_feature_extractor(
                    dL_do[0],
                    dQ_do[0],
                    episode['img_0'],
                    num_step=1,
                )

                # update target networks
                self.actor.update_target_net(self.tau_actor)
                self.critic.update_target_net(self.tau_critic)

        # plt.plot(loss)
        # plt.title('Q-function Loss')
        # plt.xlabel('iter')
        # plt.ylabel('')
        # plt.show()

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
        """ TODO """
        q = self.critic.sample_q(obs, act)
        return q

    def get_obs(self,
                img):
        """ TODO """
        obs = self.encoder.sample_obs(img)
        return obs

    def extract_episode(self,
                        dataset,
                        lstm_horizon):
        """
        """
        np.random.seed(0)
        n = dataset['img_0'].shape[0]

        good_data = False
        while not good_data:
            # idx = np.random.randint(1, n - self.lstm_horizon - 1)
            idx = np.random.randint(1, n - lstm_horizon - 1)
            print('idx:\t', idx)

            print(dataset['act'].shape)
            print(dataset['reward'].shape)

            data_pre_episode = {}
            data_pre_episode['img_0'] = dataset['img_0'][:idx]
            data_pre_episode['img_1'] = dataset['img_1'][:idx+1]
            data_pre_episode['act'] = [dataset['act'][:idx, :]]
            data_pre_episode['reward'] = [dataset['reward'][:idx]]

            episode = {}
            episode['img_0'] = dataset['img_0'][idx:idx+lstm_horizon]
            episode['img_1'] = dataset['img_1'][idx+1: idx + lstm_horizon + 1]
            episode['act'] = np.array([dataset['act'][idx:idx+lstm_horizon, :]])
            episode['reward'] = np.array([dataset['reward'][idx:idx+lstm_horizon]])

            if np.sum(episode['reward']) > 0:
                good_data = True
        return data_pre_episode, episode


def make_dummy_dataset(num_sample=100):
    # np.random.seed(0)
    imgs = np.random.randn(num_sample+1, 16, 90, 3)
    dataset = {
        'img_0': imgs[:num_sample],
        'img_1': imgs[1:num_sample+1],
        'act': np.random.randn(1, num_sample, 3),
        'reward': np.random.randn(1, num_sample, 1),
    }
    return dataset


if __name__ == "__main__":
    # np.random.seed(0)
    session = tf.compat.v1.Session()
    critic = CriticRDPG(session, )  # test_mode=True)
    actor = ActorRDPG(session, )  # test_mode=True)
    encoder = Encoder(session, )  # test_mode=True)
    agent = AgentRDPG(session, actor, critic, encoder)

    data_dir = './training_dicts/'
    fn_list = os.listdir(data_dir)

    loss_20 = []
    for i in range(120):
        # idx = np.random.randint(len(fn_list))
        idx = 0
        with open(data_dir + fn_list[idx], 'rb') as f:
            dataset = pickle.load(f)
        l = agent.train_rdpg(dataset, num_episode=4, lstm_horizon=20)
        for i in range(len(l)):
            loss_20.append(l[i])
    loss_20 = [loss_20[i] / 20 for i in range(0, len(loss_20))]

    critic = CriticRDPG(session, )  # test_mode=True)
    actor = ActorRDPG(session, )  # test_mode=True)
    encoder = Encoder(session, )  # test_mode=True)
    agent = AgentRDPG(session, actor, critic, encoder)

    loss_16 = []
    for i in range(120):
        # idx = np.random.randint(len(fn_list))
        idx = 0
        with open(data_dir + fn_list[idx], 'rb') as f:
            dataset = pickle.load(f)
        l = agent.train_rdpg(dataset, num_episode=4, lstm_horizon=16)
        for i in range(len(l)):
            loss_16.append(l[i])
    loss_16 = [loss_16[i] / 16 for i in range(0, len(loss_16))]

    critic = CriticRDPG(session, )  # test_mode=True)
    actor = ActorRDPG(session, )  # test_mode=True)
    encoder = Encoder(session, )  # test_mode=True)
    agent = AgentRDPG(session, actor, critic, encoder)

    loss_12 = []
    for i in range(120):
        # idx = np.random.randint(len(fn_list))
        idx = 0
        with open(data_dir + fn_list[idx], 'rb') as f:
            dataset = pickle.load(f)
        l = agent.train_rdpg(dataset, num_episode=4, lstm_horizon=12)
        for i in range(len(l)):
            loss_12.append(l[i])
    loss_12 = [loss_12[i] / 12 for i in range(0, len(loss_12))]

    critic = CriticRDPG(session, )  # test_mode=True)
    actor = ActorRDPG(session, )  # test_mode=True)
    encoder = Encoder(session, )  # test_mode=True)
    agent = AgentRDPG(session, actor, critic, encoder)


    loss_8 = []
    for i in range(120):
        # idx = np.random.randint(len(fn_list))
        idx = 0
        with open(data_dir + fn_list[idx], 'rb') as f:
            dataset = pickle.load(f)
        l = agent.train_rdpg(dataset, num_episode=4, lstm_horizon=8)
        for i in range(len(l)):
            loss_8.append(l[i])
    loss_8 = [loss_8[i] / 8 for i in range(0, len(loss_8))]
    critic = CriticRDPG(session, )  # test_mode=True)
    actor = ActorRDPG(session, )  # test_mode=True)
    encoder = Encoder(session, )  # test_mode=True)
    agent = AgentRDPG(session, actor, critic, encoder)

    loss_4 = []
    for i in range(120):
        # idx = np.random.randint(len(fn_list))
        idx = 0
        with open(data_dir + fn_list[idx], 'rb') as f:
            dataset = pickle.load(f)
        l = agent.train_rdpg(dataset, num_episode=4, lstm_horizon=4)
        for i in range(len(l)):
            loss_4.append(l[i])
    loss_4 = [loss_4[i] / 4 for i in range(0, len(loss_4))]
    critic = CriticRDPG(session, )  # test_mode=True)
    actor = ActorRDPG(session, )  # test_mode=True)
    encoder = Encoder(session, )  # test_mode=True)
    agent = AgentRDPG(session, actor, critic, encoder)

    plt.plot(loss_20, label="20")
    plt.legend(loc="upper right")
    plt.title('Q-function Loss')
    plt.xlabel('iter')
    plt.ylabel('')
    plt.show()

    plt.plot(loss_16, label="16")
    plt.legend(loc="upper right")
    plt.title('Q-function Loss')
    plt.xlabel('iter')
    plt.ylabel('')
    plt.show()

    plt.plot(loss_12, label="12")
    plt.legend(loc="upper right")
    plt.title('Q-function Loss')
    plt.xlabel('iter')
    plt.ylabel('')
    plt.show()

    plt.plot(loss_8, label="8")
    plt.legend(loc="upper right")
    plt.title('Q-function Loss')
    plt.xlabel('iter')
    plt.ylabel('')
    plt.show()

    plt.plot(loss_4, label="4")
    plt.legend(loc="upper right")
    plt.title('Q-function Loss')
    plt.xlabel('iter')
    plt.ylabel('')
    plt.show()

    pass
