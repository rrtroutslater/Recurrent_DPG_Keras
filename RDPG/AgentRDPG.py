from __future__ import print_function
from ActorRDPG import *
from CriticRDPG import *
from Encoder import *
import matplotlib.pyplot as plt
import os
import pickle
import math


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
                   num_episode_per_dataset=5,
                   lstm_horizon=10,
                   num_critic_step_per_actor_step=10,
                   display_act=False,
                   ):
        """
        """

        for episode in range(0, num_episode_per_dataset):

            loss = []

            data_pre_episode, episode = self.extract_episode(
                dataset,
                lstm_horizon=lstm_horizon)

            update_actor = False
            for i in range(0, num_critic_step_per_actor_step):
                # reset hidden states between training episodes
                self.actor.reset_hidden_states()
                self.critic.reset_hidden_states()

                # only update actor once per some # of critic updates
                if i == num_critic_step_per_actor_step - 1:
                    update_actor = True

                # extract observations (features) from images in dataset
                if np.array(data_pre_episode['img_0']).shape[-1] == 3:
                    obs_pre_episode = self.encoder.get_obs(
                        np.array(data_pre_episode['img_0']),
                        # display=True,
                    )
                    obs_target_pre_episode = self.encoder.get_obs(
                        np.array(data_pre_episode['img_1'])
                    )
                    obs = self.encoder.get_obs(
                        np.array(episode['img_0'])
                    )
                    obs_target = self.encoder.get_obs(
                        np.array(episode['img_1'])
                    )
                else:
                    obs_pre_episode = np.array(data_pre_episode['img_0'])
                    obs_target_pre_episode = np.array(data_pre_episode['img_1'])
                    obs = np.array(episode['img_0'])
                    obs_target = np.array(episode['img_1'])


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

                # action and Q values from target nets, used for training critic Q function
                act_target = self.actor.sample_act_target(obs_target)
                q_target = self.critic.sample_q_target(obs_target, act_target)

                # regression label: y = reward + gamma * q_target
                gamma = 0.90
                reward = np.expand_dims(reward, axis=0)
                y = np.zeros(shape=(1, reward.shape[1], 1))
                for i in range(0, reward.shape[1]):
                    y[:,i,:] = reward[:,i,:] + q_target[:,i,:] * gamma

                q_pred = self.critic.sample_q(obs, act, reset_hidden_after_sample=True)

                # get gradients from Q net, which are backpropagated through actor and encoder
                act_for_grad = self.actor.sample_act(obs, reset_hidden_after_sample=True)

                # for actor grad update
                dQ_da = self.critic.get_dQ_da_critic(obs, act_for_grad)

                # gradient step on the Q network weights
                # self.critic.get_weight_check()
                l = self.critic.train_critic(act, obs, y, num_step=1)
                # self.critic.get_weight_check()
                loss.append(l[0])

                if update_actor:
                    # print('\nUPDATING ACTOR...')
                    # self.actor.get_weight_check()
                    # grad step on actor network weights
                    self.actor.apply_gradients(obs, dQ_da, num_step=1)
                    # update actor target
                    self.actor.update_target_net(self.tau_actor)
                    # self.actor.get_weight_check()

                if display_act:
                    print('example action:\n', 
                            self.actor.sample_act(
                                obs,
                                reset_hidden_after_sample=False,
                    ))
                    print('actual action:\n', act)
                    # print(obs)
                    # print(obs[:,:,50])
                    # self.actor.test_mode = False
                    print('reward:\n', reward)
                    print('y:\n', y)
                    print('q_pred:\n', q_pred)
                    # print('idx reward, q_pred')
                    # print(np.argsort(reward[0].T[0]))
                    # print(np.argsort(q_pred[0].T[0]))
                    # self.actor.display_hidden_state()
                    display_act = False

                # update critic target
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
        with self.sess.as_default():
            act = self.actor.sample_act(obs)[0][0]
            self.actor.get_weight_check()
        print("\nact in get act:\n", act)
        if add_noise:
            act[0] += np.random.randn(1) * variance
            act[1] += np.random.randn(1) * variance / 2.0
            act[2] += np.random.randn(1) * variance / 2.0
            # act += np.random.randn(self.actor.act_dim) * variance
        return act

    def get_obs(self, img):
        with self.sess.as_default():
            obs = self.encoder.get_obs(img)
        return obs

    def get_q_pred(self,
                   obs,
                   act,
                   ):
        with self.sess.as_default():
           q = self.critic.sample_q(obs, act)
           self.critic.get_weight_check()
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
    fn_train_list = []
    # data_dir = './training_data/rlbp_data/'
    # fn_train_list = os.listdir(data_dir)
    # fn_train_list += ['./training_data/rlbp_data/'+fn for fn in os.listdir('./training_data/rlbp_data/')]
    # fn_train_list += ['./training_data/rlbp_data_2/'+fn for fn in os.listdir('./training_data/rlbp_data_2/')]
    fn_train_list += ['./training_data/rlbp_data_3/'+fn for fn in os.listdir('./training_data/rlbp_data_3/')]
    fn_train_list += ['./training_data/rlbp_data_3/'+fn for fn in os.listdir('./training_data/rlbp_data_3/')]
    fn_train_list += ['./training_data/rlbp_data_3/'+fn for fn in os.listdir('./training_data/rlbp_data_3/')]
    fn_train_list += ['./training_data/rlbp_data_4/'+fn for fn in os.listdir('./training_data/rlbp_data_4/')]
    fn_train_list += ['./training_data/rlbp_data_4/'+fn for fn in os.listdir('./training_data/rlbp_data_4/')]
    fn_train_list += ['./training_data/rlbp_data_4/'+fn for fn in os.listdir('./training_data/rlbp_data_4/')]
    fn_train_list += ['./training_data/rlbp_data_4/'+fn for fn in os.listdir('./training_data/rlbp_data_4/')]
    fn_train_list += ['./training_data/rlbp_data_4/'+fn for fn in os.listdir('./training_data/rlbp_data_4/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    fn_train_list += ['./training_data/rlbp_data_5/'+fn for fn in os.listdir('./training_data/rlbp_data_5/')]
    print(len(fn_train_list))

    actor_learning_rate = 0.0001
    critic_learning_rate = 0.0001

    actor_lstm_units = 32
    critic_lstm_units = 32

    tau_critic = 0.001
    tau_actor = 0.005
    # tau_actor = 0.01

    LSTM_HORIZON = 10

    # # untrained
    # actor_fn = None
    # actor_target_fn = None
    # critic_fn = None
    # critic_target_fn = None

    # trained 1
    # actor_fn = './trained_models/' + 'actor_102020-05-07 15:45:11.672610_10'
    # actor_target_fn = './trained_models/' + 'actor_target_10_0.00025_0.005_2020-05-07 15:45:11.672624_10'
    # critic_fn = './trained_models/' + 'critic_102020-05-07 15:45:11.454139_10'
    # critic_target_fn = './trained_models/' + 'critic_target_10_0.0001_0.001_2020-05-07 15:45:11.454174_10'
    # critic_target_fn = './trained_models/' + 'critic_target_10_0.0001_0.001_2020-05-07 15:45:11.454174_10'

    # # trained 2
    # actor_fn = './trained_models/2/' + 'actor_10_10'
    # actor_target_fn = './trained_models/2/' + 'actor_target_10_0.0001_0.005__10'
    # critic_fn = './trained_models/2/' + 'critic_10_10'
    # critic_target_fn = './trained_models/2/' + 'critic_target_10_0.0001_0.001__10'

    # trained 3
    actor_fn = './trained_models/3/' + 'actor_10_10final'
    actor_target_fn = './trained_models/3/' + 'actor_target_10_0.0001_0.005__10final'
    critic_fn = './trained_models/3/' + 'critic_10_10final'
    critic_target_fn = './trained_models/3/' + 'critic_target_10_0.0001_0.001__10final'


    session = tf.compat.v1.Session()
    tf.keras.backend.manual_variable_initialization(True)
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        tf.python.keras.backend.set_session(session)

        # initialize the encoder
        encoder = Encoder(
            session,
        )

        critic = CriticRDPG(
            session, 
            learning_rate=critic_learning_rate,
            lstm_units=critic_lstm_units,
            critic_fn=critic_fn,
            critic_target_fn=critic_target_fn,
            lstm_horizon=LSTM_HORIZON
            )

        actor = ActorRDPG(
            session, 
            learning_rate=actor_learning_rate,
            lstm_units=actor_lstm_units,
            actor_fn=actor_fn,
            actor_target_fn=actor_target_fn,
            lstm_horizon=LSTM_HORIZON
            )

        session.run(tf.compat.v1.global_variables_initializer())
        encoder.load_model(
            # pre batch normalization
            '/home/rhino/workspaces/lbp_ws/arl_subt_ws/src/exploration/lbplanner/rlbp/rlbplanner/scripts/Recurrent_DPG_Keras/RDPG/trained_encoders/'+'encoder_2020-05-04 18:40:59.673265',
            '/home/rhino/workspaces/lbp_ws/arl_subt_ws/src/exploration/lbplanner/rlbp/rlbplanner/scripts/Recurrent_DPG_Keras/RDPG/trained_encoders/'+'encoder_DECODER2020-05-04 18:40:59.673290',
        )
        encoder.get_weight_check()

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
            encoder,
            tau_actor=tau_actor,
            tau_critic=tau_critic,
            )

        loss_train = []
        display_act = False
        for i in range(0, 102):
            # training loss
            idx = np.random.randint(len(fn_train_list))
            with open(fn_train_list[idx], 'rb') as f:
                dataset = pickle.load(f)
            l = agent.train_rdpg(dataset, 
                                num_episode_per_dataset=5,
                                lstm_horizon=LSTM_HORIZON,
                                display_act=display_act
                                )
            display_act = False

            for j in range(len(l)):
                loss_train.append(l[j])

            if i % 5 == 0:
                print('\n----------------------')
                print(fn_train_list[idx])
                print('iter:\t\t', i)
                print('loss:\t\t', loss_train[-1])
                display_act = True

            if i % 50 == 0 and i > 0:
                critic_fn, critic_target_fn = critic.save_model(
                    learning_rate=str(critic_learning_rate), 
                    lstm_horizon=str(LSTM_HORIZON),
                    tau=str(tau_critic))
                actor_fn, actor_target_fn = actor.save_model(
                    learning_rate=str(actor_learning_rate), 
                    lstm_horizon=str(LSTM_HORIZON),
                    tau=str(tau_actor))


        loss_train_plot = [loss_train[i] for i in range(0, len(loss_train))]
        loss_train_plot = np.convolve(np.array(loss_train_plot), np.ones(40), 'valid') / 40
        plt.plot(loss_train_plot, label="30")
        plt.legend(loc="upper right")
        plt.title('Q-function Loss: LR: ' + str(critic_learning_rate) + " units: " + str(critic_lstm_units))
        plt.xlabel('iter')
        plt.ylabel('')
        plt.savefig("q_loss_" + "_" + str(LSTM_HORIZON) + "_" + str(critic_learning_rate) + "_" + str(critic_lstm_units) + "_" + str(i) + ".png")
        plt.close()

        critic_fn, critic_target_fn = critic.save_model(
            learning_rate=str(critic_learning_rate), 
            lstm_horizon=str(LSTM_HORIZON),
            tau=str(tau_critic))
        actor_fn, actor_target_fn = actor.save_model(
            learning_rate=str(actor_learning_rate), 
            lstm_horizon=str(LSTM_HORIZON),
            tau=str(tau_actor))

    session.close()
    del actor
    del critic
    del agent
    del session

    pass
