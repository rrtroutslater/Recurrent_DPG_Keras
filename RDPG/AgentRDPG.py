from ActorRDPG import *
from CriticRDPG import *
from Encoder import *
import matplotlib.pyplot as plt


class AgentRDPG():
    def __init__(self,
                 session,
                 actor,
                 critic,
                 encoder,
                 lstm_horizon=10,
                 gamma=0.95,
                 ):

        self.sess = session
        self.actor = actor
        self.critic = critic
        self.encoder = encoder
        self.lstm_horizon = lstm_horizon
        self.gamma = gamma

        self.learning_rate_actor = self.actor.learning_rate
        self.learning_rate_critic = self.critic.learning_rate

        return

    def train_rdpg(self,
                   dataset,
                   num_episode=20,
                   num_update=1,
                   ):
        """
        """

        loss = []
        for i in range(0, num_episode):

            data_pre_episode, episode = self.extract_episode(dataset)

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
                y = reward + gamma * q_target

                # get gradients from Q net, which are backpropagated through actor and encoder
                # NOTE: this must be done before the train_on_batch below
                # mu(o)
                act_for_grad = self.actor.sample_act(obs)
                # for actor grad update
                dQ_da = self.critic.get_dQ_da_critic(obs, act_for_grad)
                # for encoder update
                dL_do = self.critic.get_dL_do_critic(y, obs, act)
                dJ_do = self.actor.get_dJ_do_actor(obs, dQ_da)

                # take a gradient step on the Q network weights
                l = self.critic.net.train_on_batch([act, obs], y)
                loss.append(l)

                # grad step on actor
                self.actor.apply_gradients(obs, dQ_da, num_step=1)

                # grad step on encoder
                self.encoder.apply_gradients_to_feature_extractor(
                    dL_do[0],
                    dJ_do[0],
                    episode['img_0'],
                    num_step=1,
                )

                # update target networks
                self.actor.update_target_net()
                self.critic.update_target_net()

        plt.plot(loss)
        plt.title('Q-function Loss')
        plt.xlabel('iter')
        plt.ylabel('')
        plt.show()

        return

    def update_ac_hidden_state(self,
                               obs_t,
                               obs_tp1,
                               act_t,
                               act_tp1,
                               ):
        """
        propagate data prior to training episode in order to update hidden states of 
        recurrent models

        NOTE:
        act_tp1 MUST be the output of 

        inptus:
            obs_t: observation at time t, numpy array shape (N, T, obs_dim). used for:
                Q(o, a) in critic loss 
                dQ/d(mu(o)) in actor gradient step
                dmu(o)/dWa in actor gradient step
            obs_tp1: observation at time t+1, numpy array shape (N, T, obs_dim). used for:
                Q'(o', mu(o')) in critic loss (part of regression label)
            act_t: action taken at time t, numpy array shape (N, T, act_dim). used for:
                Q(o, a) in critic loss
            act_tp1: action taken at time t+1, numpy array shape (N, T, act_dim). used for:
        """
        self.actor.propagate_actor_episode(obs_t, obs_tp1)
        self.critic.propagate_critic_episode(obs_t, obs_tp1, act_t, act_tp1)
        self.critic.propagate_critic_episode(obs_t, obs_tp1, act_t)
        return

    def extract_episode(self,
                        dataset
                        ):
        """
        """
        np.random.seed(0)
        n = dataset['img_0'].shape[0]
        idx = np.random.randint(0, n - self.lstm_horizon - 1)

        data_pre_episode = {}
        data_pre_episode['img_0'] = dataset['img_0'][:idx]
        data_pre_episode['img_1'] = dataset['img_1'][:idx+1]
        data_pre_episode['act'] = dataset['act'][:, :idx, :]
        data_pre_episode['reward'] = dataset['reward'][:, :idx, :]

        episode = {}
        episode['img_0'] = dataset['img_0'][idx:idx+self.lstm_horizon]
        episode['img_1'] = dataset['img_1'][idx+1: idx + self.lstm_horizon + 1]
        episode['act'] = dataset['act'][:, idx:idx+self.lstm_horizon, :]
        episode['reward'] = dataset['reward'][:, idx:idx+self.lstm_horizon, :]
        return data_pre_episode, episode


def make_dummy_dataset(num_sample=100):
    np.random.seed(0)
    imgs = np.random.randn(num_sample+1, 16, 90, 3)
    dataset = {
        'img_0': imgs[:num_sample],
        'img_1': imgs[1:num_sample+1],
        'act': np.random.randn(1, num_sample, 3),
        'reward': np.random.randn(1, num_sample, 1),
    }
    return dataset


if __name__ == "__main__":
    np.random.seed(0)
    session = tf.compat.v1.Session()
    critic = CriticRDPG(session, )  # test_mode=True)
    actor = ActorRDPG(session, )  # test_mode=True)
    encoder = Encoder(session, )  # test_mode=True)
    agent = AgentRDPG(session, actor, critic, encoder)

    dataset = make_dummy_dataset()
    # agent.extract_episode(dataset)
    agent.train_rdpg(dataset, num_episode=4, num_update=1)

    pass
