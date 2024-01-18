import numpy as np
import tensorflow as tf
from motornet.nets.losses import PositionLoss, L2xDxRegularizer
from typing import Union
from motornet.tasks import Task
from losses import MuscleLoss, PositionLoss2

class RandomTargetReach(Task):
    #Only altered from the MotorNet version to accept a 2nd Cartesian position loss, and avoids mention of the GRU regularisation
    """A reach to a random target from a random starting position.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
            loss.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(self, network, name: str = 'RandomTargetReach', deriv_weight: float = 0., **kwargs):
        super().__init__(network, name=name, **kwargs)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = MuscleLoss(max_iso_force=max_iso_force)
        weight_loss = L2xDxRegularizer(deriv_weight=0.05, dt=dt)
        self.add_loss('excitation', loss_weight=1., loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=2.5, loss=PositionLoss())
        self.add_loss('cartesian position2', loss_weight=2.5, loss=PositionLoss2())
        for i in self.network.layer_state_names:
            self.add_loss(i, loss_weight=0.01, loss=weight_loss)


    def generate(self, batch_size, n_timesteps, validation: bool = False):
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(goal_states_j)
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs = {"inputs": targets[:, :, :self.network.plant.space_dim]}
        return [inputs, targets, init_states]


class HoldPositionWithLoads(Task):
    # Hold Position Task. Arm initialised at a random point, and acted upon by a random force
    """
    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        endpoint_load_vounds: `Float`, or `K`-items `list`, `tuple` or `numpy.ndarray`, with `K` the :attr:`space_dim`
            attribute of the :class:`motornet.plants.skeletons.Skeleton` object class or subclass, `i.e.`, the
            dimensionality of the worldspace. Gives minimum and maximum forces that can be applied to the skeleton
        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
            loss.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(
            self,
            network,
            endpoint_load_bounds: Union[float, list, tuple, np.ndarray],
            name: str = 'HoldPositionWithLoads',
            deriv_weight: float = 0.,
            **kwargs
    ):

        super().__init__(network, name=name, **kwargs)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = MuscleLoss(max_iso_force=max_iso_force)
        weight_loss = L2xDxRegularizer(deriv_weight=0.05, dt=dt)
        self.add_loss('excitation', loss_weight=1., loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=2.5, loss=PositionLoss())
        self.add_loss('cartesian position2', loss_weight=2.5, loss=PositionLoss2())
        for i in self.network.layer_state_names:
            self.add_loss(i, loss_weight=0.01, loss=weight_loss)
        self.endpoint_load_param = endpoint_load_bounds

    def generate(self, batch_size, n_timesteps, validation: bool = False):
        
        init_states = self.get_initial_state(batch_size=batch_size)
        goal_states = self.network.plant.joint2cartesian(init_states[0][:, :])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        endpoint_force = np.repeat(np.random.uniform(low = self.endpoint_load_param[0], high = self.endpoint_load_param[1], size = (batch_size,2)),n_timesteps, axis = 0)
        endpoint_load = tf.constant(endpoint_force, shape=(batch_size, n_timesteps, 2))
        
        inputs = {"inputs": targets[:, :, :self.network.plant.space_dim], "endpoint_load": endpoint_load}
        return [inputs, targets, init_states]