
    # def apply_gradients_to_Q_net(self,
    #         act,
    #         obs,
    #         num_step=1,
    #     ):
    #     """
    #     compute and apply gradient of Bellman Error w.r.t. Q network weights

    #     inputs:
    #         dJ_da: gradient of loss with respect to Q value, numpy array, 
    #             shape (N, num_timestep, 1)
    #         act: series of actions, numpy array,
    #             shape (N, num_timestep, act_dim)
    #         obs: series of observations, numpy array, 
    #             shape (N, num_timestep, obs_dim)
    #         num_step: number of gradient steps to perform
    #     """
    #     for i in range(0, num_step):
    #         if self.test_mode:
    #             print('----------\nweights before update:', self.net_weights[0].eval(session=self.sess))
    #         self.sess.run(
    #             self.grad_step,
    #             feed_dict={
    #                 # self.dL_dq: dL_dq,
    #                 self.act_in: act,
    #                 self.obs_in: obs,
    #             }
    #         )
    #         if self.test_mode:
    #             print('----------\nweights after update:', self.net_weights[0].eval(session=self.sess))
    #     return

            # placeholder for Q training labels
        # self.label_q_ph = tf.keras.backend.placeholder(shape=[None, None, 1])
        # self.label_q_ph = tf.keras.backend.placeholder(shape=[None, 1])

        # loss operation
        # print(self.q.get_shape())
        # self.loss = tf.keras.losses.MeanSquaredError()


        # training for Q function
        # self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.loss, var_list=self.net_weights)
        # self.grad_step = self.optimizer.apply_gradients(zip(self.dL_dWq, self.net_weights))