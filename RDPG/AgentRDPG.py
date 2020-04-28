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
                   dataset_is_validation=False,
                   display_act=False,
                   ):
        """
        """
        loss = []
        validation_loss = []

        # reset hidden states between training episodes
        self.actor.reset_hidden_states()
        self.critic.reset_hidden_states()

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
        self.actor.sample_act(obs_pre_episode)  # actor hidden state update
        act_target_pre_episode = self.actor.sample_act_target(obs_target_pre_episode) # actor target hidden state update
        act_pre_episode = data_pre_episode['act']

        # 2) use actions and obs before episode to update Q network hidden states
        q_pre_episode = self.critic.sample_q(obs_pre_episode, act_pre_episode)  # Q update
        q_target_pre_episode = self.critic.sample_q_target(obs_target_pre_episode, act_target_pre_episode) # Q target update

        # actions, rewards which occurred during episode
        act = episode['act']
        reward = episode['reward'].T

        # probably can't do more than a single update (to critic?) without re-initializing hidden states
        for i in range(0, num_update):
            # action and Q values from target nets, used for training critic Q function
            act_target = self.actor.sample_act_target(obs_target)
            q_target = self.critic.sample_q_target(obs_target, act_target)

            # regression label: y = reward + gamma * q_target
            gamma = 0.99
            reward = np.expand_dims(reward, axis=0)
            y = np.zeros(shape=(1, reward.shape[1], 1))
            for i in range(0, reward.shape[1]):
                y[:,i,:] = reward[:,i,:] + q_target[:,i,:] * gamma

            q_pred = self.critic.sample_q(obs, act, reset_hidden_after_sample=True)

            # get gradients from Q net, which are backpropagated through actor and encoder
            act_for_grad = self.actor.sample_act(obs, reset_hidden_after_sample=True)

            # for actor grad update
            dQ_da = self.critic.get_dQ_da_critic(obs, act_for_grad)

            if not dataset_is_validation:
                # gradient step on the Q network weights
                l = self.critic.train_critic(act, obs, y, num_step=1)
                loss.append(l[0])

                # grad step on actor network weights
                self.actor.apply_gradients(obs, dQ_da, num_step=1)

                # update target networks
                self.actor.update_target_net(self.tau_actor)
                self.critic.update_target_net(self.tau_critic)
            else:
                # get validation loss, do not train
                l_val = self.critic.get_loss(act, np.array(obs), y)
                validation_loss.append(l_val[0])

            if display_act:
                # self.actor.test_mode = True
                # plt.imshow(o[0][:,:,0])
                # plt.show()
                print('example action:\n', 
                        self.actor.sample_act(
                            obs,
                            reset_hidden_after_sample=False,
                ))
                print('actual action:\n', act)
                # self.actor.test_mode = False
                print('reward:\n', reward)
                print('y:\n', y)
                print('q_pred:\n', q_pred)
                print('idx reward, q_pred')
                print(np.argsort(reward[0].T[0]))
                print(np.argsort(q_pred[0].T[0]))

        return loss, validation_loss

    def get_act(self,
                obs,
                add_noise=False,
                variance=2):
        """ 
        TODO 
        use heavy-tail distribution instead of gaussian
        """
        act = self.actor.sample_act(obs)[0][0]
        print("\nact in get act:\n", act)
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
    # data_dir = './training_dicts_UNTRAINED/'
    data_dir = './training_dicts_TRAINED_1/'
    fn_list = os.listdir(data_dir)

    fn_train_list = np.random.choice(fn_list, int(0.95 * len(fn_list)), replace=False)
    fn_val_list = []
    for fn in fn_list:
        if fn not in fn_train_list:
            fn_val_list.append(fn)
    print('training sets:\t\t', len(fn_train_list))
    print('validation sets:\t', len(fn_val_list))

    # actor_learning_rate = 0.000001
    actor_learning_rate = 0.00001
    critic_learning_rate = 0.0005

    # actor_lstm_units = 64
    actor_lstm_units = 32
    critic_lstm_units = 64

    tau = 0.001
    # tau = 0.05
    # tau = 0.2

    # lstm_horizons = [30, 20, 10, 5]
    lstm_horizons = [10, 20]
    # LSTM_HORIZON = 30

    for LSTM_HORIZON in lstm_horizons:
        actor_fn = None
        actor_target_fn = None
        critic_fn = None
        critic_target_fn = None
        loss_train = []

        for k in range(0, 1):

            session = tf.compat.v1.Session()
            tf.keras.backend.manual_variable_initialization(True)
            graph = tf.compat.v1.get_default_graph()

            display_act=False

            with graph.as_default():
                tf.python.keras.backend.set_session(session)

                critic = CriticRDPG(
                    session, 
                    learning_rate=critic_learning_rate,
                    lstm_units=critic_lstm_units,
                    critic_fn=critic_fn,
                    critic_target_fn=critic_target_fn,
                    )

                actor = ActorRDPG(
                    session, 
                    learning_rate=actor_learning_rate,
                    lstm_units=actor_lstm_units,
                    actor_fn=actor_fn,
                    actor_target_fn=actor_target_fn,
                    )

                session.run(tf.compat.v1.global_variables_initializer())

                if actor_fn is not None:
                    actor.load_model(actor_fn, actor_target_fn)
                    critic.load_model(critic_fn, critic_target_fn)
                else:
                    actor.update_target_net(copy_all=True)
                    critic.update_target_net(copy_all=True)

                agent = AgentRDPG(
                    session, 
                    actor, 
                    critic,
                    tau_actor=tau,
                    tau_critic=tau,
                    )

                for i in range(0, 60000):
                    # training loss
                    idx = np.random.randint(len(fn_train_list))
                    with open(data_dir + fn_train_list[idx], 'rb') as f:
                        dataset = pickle.load(f)
                    l, _ = agent.train_rdpg(dataset, 
                                        num_episode_per_dataset=1,
                                        lstm_horizon=LSTM_HORIZON,
                                        num_update=1,
                                        dataset_is_validation=False,
                                        display_act=display_act
                                        )
                    display_act = False

                    for j in range(len(l)):
                        loss_train.append(l[j])

                    if i % 100 == 0:
                        print('\n----------------------')
                        print(fn_train_list[idx])
                        print('iter:\t\t', i)
                        print('loss:\t\t', loss_train[-1])
                        display_act=True

                    if i % 5000 == 0 and i > 0:
                        loss_train_plot = [loss_train[i] for i in range(0, len(loss_train))]
                        loss_train_plot = np.convolve(np.array(loss_train_plot), np.ones(40), 'valid') / 40
                        plt.plot(loss_train_plot, label="30")
                        plt.legend(loc="upper right")
                        plt.title('Q-function Loss: LR: ' + str(critic_learning_rate) + ' tau: ' + str(tau) + "units: " + str(critic_lstm_units))
                        plt.xlabel('iter')
                        plt.ylabel('')
                        plt.savefig("q_loss_" + "_" + str(LSTM_HORIZON) + "_" + str(critic_learning_rate) + "_" + str(critic_lstm_units) + "_" + str(i) + ".png")
                        plt.close()

                        critic_fn, critic_target_fn = critic.save_model(
                            learning_rate=str(critic_learning_rate), 
                            lstm_horizon=str(LSTM_HORIZON),
                            tau=str(i))
                        actor_fn, actor_target_fn = actor.save_model(
                            learning_rate=str(actor_learning_rate), 
                            lstm_horizon=str(LSTM_HORIZON),
                            tau=str(i))

            session.close()
            del actor
            del critic
            del agent
            del session

    pass
















