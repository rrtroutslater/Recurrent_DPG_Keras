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




"""




class AgentRDPG():
    def __init__(self,
            actor,
            critic):

        return

if __name__ == "__main__":

    critic = CriticRDPG()
    actor = ActorRDPG()
    agent = AgentRDPG(actor, critic)
    actor.print_network_info()

    pass