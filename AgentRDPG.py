from ActorRDPG import *
from CriticRDPG import *

"""
TODO:

- do we need to pass a tensorflow session at all?
    * try without!

- model definition

- parameters class for each?

- create actor and critic models

- required gradients...

the "q_grad" is the derivative of the loss (bellman) with respect to action
dL/da
a = mu(f), where f is feature

derivative of loss w.r.t. feature:
dmu/df * dL/da 





"""




class AgentRDPG():
    def __init__(self,
            actor,
            critic):

        return

if __name__ == "__main__":
    session = tf.compat.v1.Session()
    critic = CriticRDPG(session)
    actor = ActorRDPG(session)
    agent = AgentRDPG(actor, critic)
    actor.print_network_info()
    actor.export_model_figure()

    feature = []
    for i in range(0, 4):
        feature.append(np.random.randn(32))
    feature = np.array(feature)
    feature = np.expand_dims(feature, axis=0)
    
    print(feature.shape)
    q_grad = []
    for i in range(0,4):
        q_grad.append(np.random.randn(3))
    q_grad = np.array(q_grad)
    q_grad = np.expand_dims(q_grad, axis=0)

    actor.apply_gradients(
        feature,
        q_grad,
        num_step=1
    )

    print(actor.sample_act(feature))
    print(actor.sample_act_target(feature))
    actor.update_target_net()
    pass