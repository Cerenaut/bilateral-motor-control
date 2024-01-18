import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense, Lambda
from abc import abstractmethod, ABC
from typing import Union
from motornet.utils import Alias
import json
import os
from keras.src.engine import compile_utils
import copy

class Network2(Layer):
    # This is an alteration to the Network class in MotorNet, but allowing for a 2nd Cartesian loss (cartesian2) to be used
    """Base class for controller :class:`Network` objects. It implements a network whose function is to control the
    plant provided as input at initialization. This object can be subclassed to implement virtually anything that
    `tensorflow` can implement as a deep neural network, so long as it abides by the state structure used in `motornet`
    (see below for details).

    Args:
        plant: A :class:`motornet.plants.plants.Plant` object class or subclass. This is the plant that the
            :class:`Network` will control.
        proprioceptive_noise_sd: `Float`, the standard deviation of the gaussian noise process for the proprioceptive
            feedback loop. The gaussian noise process is a normal distribution centered on `0`.
        visual_noise_sd: `Float`, the standard deviation of the random noise process for the visual
            feedback loop. The random process is a normal distribution centered on `0`.
        n_ministeps: `Integer`, the number of timesteps that the plant is simulated forward for each forward pass of
            the deep neural network. For instance, if the (global) timestep size is `1` ms, and `n_ministeps` is `5`,
            then the plant will be simulated for every `0.2` ms timesteps, with the excitatory drive from the controller
            only being updated every `1` ms.
        **kwargs: This is passed to the parent `tensorflow.keras.layers.Layer` class as-is.
    """

    def __init__(self, plant, proprioceptive_noise_sd: float = 0., visual_noise_sd: float = 0., n_ministeps: int = 1,
                 **kwargs):

        # set noise levels
        self.proprioceptive_noise_sd = proprioceptive_noise_sd
        self.visual_noise_sd = visual_noise_sd

        # plant states
        self.proprioceptive_delay = plant.proprioceptive_delay
        self.visual_delay = plant.visual_delay
        self.n_muscles = plant.n_muscles
        self.state_size = [
            tf.TensorShape([plant.output_dim]),
            tf.TensorShape([plant.output_dim]),
            tf.TensorShape([plant.muscle_state_dim, self.n_muscles]),
            tf.TensorShape([plant.geometry_state_dim, self.n_muscles]),
            tf.TensorShape([self.n_muscles * 2, self.proprioceptive_delay]),  # muscle length & velocity
            tf.TensorShape([plant.space_dim, self.visual_delay]),
            tf.TensorShape([plant.input_dim]),
        ]
        self.initial_state_names = [
            'joint0',
            'cartesian0',
            'cartesian2',
            'muscle0',
            'geometry0',
            'proprio_feedback0',
            'visual_feedback0',
            'excitation',
        ]
        self.output_names = [
            'joint position',
            'cartesian position',
            'cartesian position2',
            'muscle state',
            'geometry state',
            'proprioceptive feedback',
            'visual feedback',
            'excitation'
        ]

        # create attributes
        self.n_ministeps = int(np.maximum(n_ministeps, 1))
        self.output_size = self.state_size
        self.plant = plant
        self.layers = []

        # functionality for recomputing inputs at every timestep
        self.do_recompute_inputs = False
        self.recompute_inputs = lambda inputs, states: inputs

        # create Lambda-wrapped functions (to prevent memory leaks)
        def get_new_proprio_feedback(mstate):
            # normalise by muscle characteristics
            muscle_len = tf.slice(mstate, [0, 1, 0], [-1, 1, -1]) / self.plant.muscle.l0_ce
            muscle_vel = tf.slice(mstate, [0, 2, 0], [-1, 1, -1]) / self.plant.muscle.vmax
            # flatten muscle length and velocity
            proprio_true = tf.reshape(tf.concat([muscle_len, muscle_vel], axis=1), shape=(-1, self.n_muscles * 2))
            return proprio_true

        def get_new_visual_feedback(cstate):
            visual_true, _ = tf.split(cstate, 2, axis=-1)  # position only (discard velocity)
            return visual_true

        name = "get_new_hidden_state"
        self.get_new_hidden_state = Lambda(lambda x: [tf.zeros((x[0], n), dtype=x[1]) for n in self.n_units], name=name)
        self.unpack_plant_states = Lambda(lambda x: x[:4], name="unpack_plant_states")
        self.unpack_feedback_states = Lambda(lambda x: x[4:6], name="unpack_feedback_states")
        self.get_feedback_backlog = Lambda(lambda x: tf.slice(x, [0, 0, 1], [-1, -1, -1]), name="get_feedback_backlog")
        self.get_feedback_current = Lambda(lambda x: x[:, :, 0], name="get_feedback_current")
        self.lambda_cat = Lambda(lambda x: tf.concat(x, axis=-1), name="lambda_cat")
        self.lambda_cat2 = Lambda(lambda x: tf.concat(x, axis=2), name="lambda_cat2")
        self.add_noise = Lambda(lambda x: x[0] + tf.random.normal(tf.shape(x[0]), stddev=x[1]), name="add_noise")
        self.tile_feedback = Lambda(lambda x: tf.tile(x[0][:, :, tf.newaxis], [1, 1, x[1]]), name="tile_feedback")
        self.get_new_proprio_feedback = Lambda(lambda x: get_new_proprio_feedback(x), name="get_new_proprio_feedback")
        self.get_new_visual_feedback = Lambda(lambda x: get_new_visual_feedback(x), name="get_new_visual_feedback")
        self.get_new_excitation_state = Lambda(lambda x: tf.zeros((x[0], self.plant.input_dim), dtype=x[1]))
        self.built = False

        super().__init__(**kwargs)

    state_name = Alias("output_names", alias_name="state_name")
    """An alias name for the `output_names` attribute."""

    @abstractmethod
    def forward_pass(self, inputs, states):
        """Performs the forward pass through the network layers to obtain the motor commands that will then be passed
        on to the plant.

        Args:
            inputs: `Tensor`, inputs to the first layer of the network.
            states: `List` of `tensor` arrays, containing the states of each layer operating on a state.

        Returns:
            - A `tensor` array, the output of the last layer to use as the motor command, or excitation to the plant.
            - A `list` of the new states inherent to potential layers operating on a state.
            - A `dictionary` of the new states inherent to potential layers operating on a state.

        Raises:
            NotImplementedError: If this method is not overwritten by a subclass object.
        """
        raise NotImplementedError("This method must be overwritten by a subclass object.")

    def get_base_config(self):
        """Gets the object instance's base configuration. This is the set of configuration entries that will be useful
        for any :class:`Network` class or subclass. This method should be called by the :meth:`get_save_config`
        method. Users wanting to save additional configuration entries specific to a `Network` subclass should then
        do so in the :meth:`get_save_config` method, using this method's output `dictionary` as a base.

        Returns:
             A `dictionary` containing the network's proprioceptive and visual noise standard deviation and delay, and
             the number of muscles and ministeps.
        """

        cfg = {
            'proprioceptive_noise_sd': self.proprioceptive_noise_sd,
            'visual_noise_sd': self.visual_noise_sd,
            'proprioceptive_delay': self.proprioceptive_delay,
            'visual_delay': self.visual_delay,
            'n_muscle': self.n_muscles,
            'n_ministeps': self.n_ministeps
        }
        return cfg

    def get_save_config(self):
        """Gets the :class:`Network` object's configuration as a `dictionary`. This method should be overwritten by
        subclass objects, and used to add configuration entries specific to that subclass.

        Returns:
            By default, this method returns the output of the :meth:`get_base_config` method.
        """
        return self.get_base_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, states=None, **kwargs):
        """The logic for a single simulation step. This performs a single forward pass through the network, and passes
        the network output as excitation signals (motor commands) to the plant object to simulate movement.

        Args:
            inputs: `Dictionary` of `tensor` arrays. At the very least, this should contain a "inputs" key mapped to
                a `tensor` array, which will be passed as-is to the network's input layer. Additional keys will be
                passed as `**kwargs` to the plant call.
            states: `List`, contains all the states of the plant, and of the network if any exist. The state order in
                the `list` follows the same convention as the :meth:`get_initial_states` method.
            kwargs: For backward compatibility only.

        Returns:
            - A `dictionary` containing the new states.
            - A `list` containing the new states in the same order convention as the :meth:`get_initial_states` method.
              While this output is redundant to the user, it is necessary for `tensorflow` to process the network over
              time.
        """

        # handle feedback
        old_proprio_feedback, old_visual_feedback = self.unpack_feedback_states(states)
        proprio_backlog = self.get_feedback_backlog(old_proprio_feedback)
        visual_backlog = self.get_feedback_backlog(old_visual_feedback)
        proprio_fb = self.get_feedback_current(old_proprio_feedback)
        visual_fb = self.get_feedback_current(old_visual_feedback)

        # if the task demands it, inputs will be recomputed at every timestep
        if self.do_recompute_inputs:
            inputs = self.recompute_inputs(inputs, states)

        x = self.lambda_cat((proprio_fb, visual_fb, inputs.pop("inputs")))
        u, new_network_states, new_network_states_dict = self.forward_pass(x, states)

        # plant forward pass
        jstate, cstate, mstate, gstate = self.unpack_plant_states(states)
        for _ in range(self.n_ministeps):
            jstate, cstate, mstate, gstate = self.plant(u, jstate, mstate, gstate, **inputs)

        proprio_true = self.get_new_proprio_feedback(mstate)
        visual_true = self.get_new_visual_feedback(cstate)
        proprio_noisy = self.add_noise((proprio_true, self.proprioceptive_noise_sd))
        visual_noisy = self.add_noise((visual_true, self.visual_noise_sd))
        new_proprio_feedback = self.lambda_cat2((proprio_backlog, proprio_noisy[:, :, tf.newaxis]))
        new_visual_feedback = self.lambda_cat2((visual_backlog, visual_noisy[:, :, tf.newaxis]))

        # pack new states
        new_states = [jstate, cstate, mstate, gstate, new_proprio_feedback, new_visual_feedback, u]
        new_states.extend(new_network_states)

        # pack output
        output = {
            'joint position': jstate,
            'cartesian position': cstate,
            'cartesian position2': cstate,
            'muscle state': mstate,
            'geometry state': gstate,
            'proprioceptive feedback': new_proprio_feedback,
            'visual feedback': new_visual_feedback,
            'excitation': u,
            **new_network_states_dict
        }

        return output, new_states

    def get_base_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        """Creates the base initial states for the first timestep of the network training procedure. This method
        provides the base states for the default :class:`Network` class, in the order listed below:

            - joint state
            - cartesian state
            - muscle state
            - geometry state
            - proprioception feedback array
            - visual feedback array
            - excitation state

        This method should be called in the :meth:`get_initial_state` method to provide a base for the output of that
        method.

        Args:
            inputs: The joint state from which the other state values are inferred. This is passed as-is to the
                :meth:`motornet.plants.plants.Plant.get_initial_state` method, and therefore obeys the structure
                documented there.
            batch_size: `Integer`, the batch size defining the size of each state's first dimension.
            dtype: A `dtype` from the `tensorflow.dtypes` module.

        Returns:
            A `list` of the states as `tensor` arrays in the order listed above.
        """

        if inputs is not None:
            states = self.plant.get_initial_state(joint_state=inputs, batch_size=batch_size)
        else:
            states = self.plant.get_initial_state(batch_size=batch_size)

        # no need to add noise as this is just a placeholder for initialization purposes (i.e., not used in first pass)
        excitation = self.get_new_excitation_state((batch_size, dtype))

        proprio_true = self.get_new_proprio_feedback(states[2])
        visual_true = self.get_new_visual_feedback(states[1])
        proprio_tiled = self.tile_feedback((proprio_true, self.proprioceptive_delay))
        visual_tiled = self.tile_feedback((visual_true, self.visual_delay))
        proprio_noisy = self.add_noise((proprio_tiled, self.proprioceptive_noise_sd))
        visual_noisy = self.add_noise((visual_tiled, self.visual_noise_sd))

        states.append(proprio_noisy)
        states.append(visual_noisy)
        states.append(excitation)
        return states

    def get_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        """Creates the initial states for the first timestep of the network training procedure. This method
        provides the states for the full :class:`Network` class, that is the default states from the
        :meth:`get_base_initial_state` method followed by the states specific to a potential subclass. This method
        should be overwritten by subclassing objects, and used to add new states specific to that subclass.

        Args:
            inputs: The joint state from which the other state values are inferred. This is passed as-is to the
                :meth:`motornet.plants.plants.Plant.get_initial_state` method, and therefore obeys the logic
                documented there.
            batch_size: `Integer`, the batch size defining the size of each state's first dimension.
            dtype: A `dtype` from the `tensorflow.dtypes` module.

        Returns:
            By default, this method returns the output of the :meth:`get_base_initial_state` method, that is a
            `list` of the states as `tensor` arrays.
        """
        return self.get_base_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)

class unilateral(Network2):
    
    # Our baline unilateral network architecture
    """
    Args:
        plant: A :class:`motornet.plants.plants.Plant` object class or subclass. This is the plant that the `Network`
            will control.
        n_units: `Integer` or `list`, the number of neurons per layer. If only one layer is created, then this can
            be an `integer`.
        n_hidden_layers: `Integer`, the number of hidden layers that the network will implement.
        activation: `String` or activation function from `tensorflow`. The activation function used as non-linearity
            for all GRUs.
        kernel_regularizer: `Float`, the kernel regularization weight for the hidden layers.
        recurrent_regularizer: `Float`, the recurrent regularization weight for the hidden layers.
        hidden_noise_sd: `Float`, the standard deviation of the gaussian noise process applied to GRU hidden activity.
        output_bias_initializer: A `tensorflow.keras.initializers` instance to initialize the biases of the
            last layer of the network (`i.e.`, the output layer).
        output_kernel_initializer: A `tensorflow.keras.initializers` instance to initialize the kernels of the
            last layer of the network (`i.e.`, the output layer).
        **kwargs: This is passed to the parent `tensorflow.keras.layers.Layer` class as-is.
    """

    def __init__(self, plant, n_units: Union[int, list] = 20, n_hidden_layers: int = 1, activation='tanh', **kwargs):

        super().__init__(plant, **kwargs)

        if type(n_units) == int:
            n_units = list(np.repeat(n_units, n_hidden_layers).astype('int32'))
        if len(n_units) > 1 and n_hidden_layers == 1:
            n_hidden_layers = len(n_units)
        if len(n_units) != n_hidden_layers:
            raise ValueError('The number of hidden layers should match the size of the n_unit array.')

        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        self.layer_state_names = ['hidden_' + str(k) for k in range(self.n_hidden_layers)]
        self.output_names.extend(self.layer_state_names)
        self.initial_state_names.extend([name + '_0' for name in self.layer_state_names])

        if activation == 'recttanh':
            self.activation = recttanh
            self.activation_name = 'recttanh'
        else:
            self.activation = activation
            self.activation_name = activation

    def build(self, input_shapes):

        for k in range(self.n_hidden_layers):
            layer = Dense(
                units=self.n_units[k],
                activation=self.activation,
                name='hidden_layer_' + str(k),
            )
            self.layers.append(layer)

        output_layer = Dense(
            units=self.plant.input_dim,
            activation='sigmoid',
            name='output_layer',
        )

        self.layers.append(output_layer)
        self.built = True

    def get_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        """Creates the initial states for the first timestep of the network training procedure. This method
        provides the states for the full :class:`Network` class, that is the default states from the
        :meth:`Network.get_base_initial_state` method followed by the states specific to this subclass.

        Args:
            inputs: The joint state from which the other state values are inferred. This is passed as-is to the
                :meth:`motornet.plants.plants.Plant.get_initial_state` method, and therefore obeys the structure documented
                there.
            batch_size: `Integer`, the batch size defining the size of each state's first dimension.
            dtype: A `dtype` from the `tensorflow.dtypes` module.

        Returns:
            A `list` containing the output of the :meth:`Network.get_base_initial_state` method
        """
        states = self.get_base_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)
        #hidden_states = self.get_new_hidden_state((batch_size, dtype))
        #states.extend(hidden_states)
        return states

    def get_save_config(self):
        base_config = self.get_base_config()
        cfg = {
            'n_units': int(self.n_units[0]),
            'n_hidden_layers': self.n_hidden_layers,
            'activation': self.activation_name, **base_config
        }
        return cfg

    @tf.function
    def forward_pass(self, inputs, states):
        """Performs the forward pass computation.

        Args:
            inputs: `Tensor`, inputs to the first layer of the network.
            states: `List` of `tensor` arrays representing the states of each layer operating on a state (state-based
                layers).

        Returns:
            - A `tensor` array, the output of the last layer to use as the motor command, or excitation to the plant.
            - A `list` of the new hidden states of the GRU layers.
            - A `dictionary` of the new hidden states of the GRU layers.
        """
        new_hidden_states_dict = {}
        new_hidden_states = []
        x = inputs

        for k in range(self.n_hidden_layers):
            x = self.layers[k](x)
        u = self.layers[-1](x)
        #The hidden states here are an artifact of the code I based this off using GRU layers instead of dense layers
        return u, new_hidden_states, new_hidden_states_dict

class Wt_Add(tf.keras.layers.Layer):
    # weighted addition layer used for combining the 2 hemispheres of the bilateral models
    # weights are trainable and initialised to 0.5, 0.5
    def __init__(self):
        super(Wt_Add, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w1 = tf.Variable(
            initial_value=0.5,
            trainable=True,
            constraint=lambda z: tf.nn.relu(z),
            name = "wtAdd_w1"
            )
        self.w2 = tf.Variable(
            initial_value=0.5,
            trainable=True,
            constraint=lambda z: tf.nn.relu(z),
            name = "wtAdd_w2"
            )      

    def call(self, input1, input2):
        return tf.multiply(input1,self.w1) + tf.multiply(input2, self.w2)

class bilateral_wtAdd_lesioned(Network2):
    #Our bilateral network architecture
    
    """
    Args:
        plant: A :class:`motornet.plants.plants.Plant` object class or subclass. This is the plant that the `Network`
            will control.
        n_units: `Integer` or `list`, the number of neurons per layer. If only one layer is created, then this can
            be an `integer`.
        n_hidden_layers: `Integer`, the number of hidden layers that the network will implement.
        activation: `String` or activation function from `tensorflow`. The activation function used as non-linearity
            for hidden layers.
        kernel_regularizer: `Float`, the kernel regularization weight for the hidden layers.
        recurrent_regularizer: `Float`, the recurrent regularization weight for the hidden layers.
        output_bias_initializer: A `tensorflow.keras.initializers` instance to initialize the biases of the
            last layer of the network (`i.e.`, the output layer).
        output_kernel_initializer: A `tensorflow.keras.initializers` instance to initialize the kernels of the
            last layer of the network (`i.e.`, the output layer).
        lesion: Lesion used in testing to easily lesion the model (options None, highleft, highright, lowleft, lowright)
        **kwargs: This is passed to the parent `tensorflow.keras.layers.Layer` class as-is.
    """

    def __init__(self, plant, n_units: Union[int, list] = 20, n_hidden_layers: int = 1, activation='tanh',lesion = None, **kwargs):

        super().__init__(plant, **kwargs)

        if type(n_units) == int:
            n_units = list(np.repeat(n_units, n_hidden_layers).astype('int32'))
        if len(n_units) > 1 and n_hidden_layers == 1:
            n_hidden_layers = len(n_units)
        if len(n_units) != n_hidden_layers:
            raise ValueError('The number of hidden layers should match the size of the n_unit array.')

        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        self.layer_state_names = []
        for k in range(self.n_hidden_layers):
          self.layer_state_names.append('hidden_l_' + str(k))
          self.layer_state_names.append('hidden_r_' + str(k))
        self.output_names.extend(self.layer_state_names)
        self.initial_state_names.extend([name + '_0' for name in self.layer_state_names])


        self.lesion = lesion

        if activation == 'recttanh':
            self.activation = recttanh
            self.activation_name = 'recttanh'
        else:
            self.activation = activation
            self.activation_name = activation
        
        # loss_dict used for specialised training to keep track of which layers should be trained by which loss function. 
        # Irrelevant for Non-specialised models
        self.loss_dict = {}

    def build(self, input_shapes):
        
        for k in range(self.n_hidden_layers):
            layer_l = Dense(
                units=self.n_units[k],
                activation=self.activation,
                name='hidden_l_' + str(k)
            )
            self.loss_dict['hidden_l_' + str(k)  + '/kernel:0'] = 0
            self.loss_dict['hidden_l_' + str(k)  + '/bias:0'] = 0
            self.layers.append(layer_l)

            layer_r = Dense(
                units=self.n_units[k],
                activation=self.activation,
                name='hidden_r_' + str(k)
            )
            self.loss_dict['hidden_r_' + str(k)  + '/kernel:0'] = 1
            self.loss_dict['hidden_r_' + str(k)  + '/bias:0'] = 1
            self.layers.append(layer_r)

        combine_layer = Wt_Add()
        for v in combine_layer.trainable_variables:
                self.loss_dict[v.name] = -1

        output_layer = Dense(
            units=self.plant.input_dim,
            activation='sigmoid',
            name='output_layer'
        )
        self.loss_dict['output_layer/kernel:0'] = -1
        self.loss_dict['output_layer/bias:0'] = -1

        self.layers.append(combine_layer)
        self.layers.append(output_layer)
        self.built = True

    def get_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        states = self.get_base_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)
        #hidden_states = self.get_new_hidden_state((batch_size, dtype))
        #states.extend(hidden_states)
        return states

    def get_save_config(self):
        base_config = self.get_base_config()
        cfg = {
            'n_units': int(self.n_units[0]),
            'n_hidden_layers': self.n_hidden_layers,
            'activation': self.activation_name, **base_config
        }
        return cfg

    @tf.function
    def forward_pass(self, inputs, states):
        """Performs the forward pass computation.

        Args:
            inputs: `Tensor`, inputs to the first layer of the network.
            states: `List` of `tensor` arrays representing the states of each layer operating on a state (state-based
                layers).

        Returns:
            - A `tensor` array, the output of the last layer to use as the motor command, or excitation to the plant.
            - A `list` of the new hidden states of the GRU layers.
            - A `dictionary` of the new hidden states of the GRU layers.
        """
        new_hidden_states_dict = {}
        new_hidden_states = []
        xl = inputs
        xr = inputs

        if(self.lesion == "highleft"):
            xl = tf.multiply(xl,0)
        if(self.lesion == "highright"):
            xr = tf.multiply(xl,0)

        for k in range(self.n_hidden_layers):

            l_idx = k*2
            xl = self.layers[l_idx](xl)

            r_idx = (k*2) + 1
            xr = self.layers[r_idx](xr)

        
        if(self.lesion == "lowleft"):
            xl = tf.multiply(xl,0)
        if(self.lesion == "lowright"):
            xr = tf.multiply(xr,0)

        x = self.layers[-2](xl,xr)

        u = self.layers[-1](x)
        return u, new_hidden_states, new_hidden_states_dict

class bilateral_wtAdd_CC_lesioned(Network2):
    
    # Our Corpus Callosum model architecture
    """
    Args:
        plant: A :class:`motornet.plants.plants.Plant` object class or subclass. This is the plant that the `Network`
            will control.
        n_units: `Integer` or `list`, the number of neurons per layer. If only one layer is created, then this can
            be an `integer`.
        n_hidden_layers: `Integer`, the number of hidden layers that the network will implement.
        activation: `String` or activation function from `tensorflow`. The activation function used as non-linearity
            for hidden layers.
        kernel_regularizer: `Float`, the kernel regularization weight for the hidden layers.
        recurrent_regularizer: `Float`, the recurrent regularization weight for the hidden layers.
        output_bias_initializer: A `tensorflow.keras.initializers` instance to initialize the biases of the
            last layer of the network (`i.e.`, the output layer).
        output_kernel_initializer: A `tensorflow.keras.initializers` instance to initialize the kernels of the
            last layer of the network (`i.e.`, the output layer).
        lesion: Lesion used in testing to easily lesion the model (options None, highleft, highright, lowleft, lowright, cc, ccl, ccr)
        **kwargs: This is passed to the parent `tensorflow.keras.layers.Layer` class as-is.
    """

    def __init__(self, plant, n_units: Union[int, list] = 20, n_hidden_layers: int = 1, activation='tanh',lesion = None, **kwargs):

        super().__init__(plant, **kwargs)

        if type(n_units) == int:
            n_units = list(np.repeat(n_units, n_hidden_layers).astype('int32'))
        if len(n_units) > 1 and n_hidden_layers == 1:
            n_hidden_layers = len(n_units)
        if len(n_units) != n_hidden_layers:
            raise ValueError('The number of hidden layers should match the size of the n_unit array.')

        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        self.layer_state_names = []
        for k in range(self.n_hidden_layers):
          self.layer_state_names.append('hidden_l_' + str(k))
          self.layer_state_names.append('hidden_r_' + str(k))
        self.output_names.extend(self.layer_state_names)
        self.initial_state_names.extend([name + '_0' for name in self.layer_state_names])

        self.lesion = lesion

        #for n in n_units:
        #    self.state_size.append(tf.TensorShape([n]))

        if activation == 'recttanh':
            self.activation = recttanh
            self.activation_name = 'recttanh'
        else:
            self.activation = activation
            self.activation_name = activation
        
        # loss_dict used for specialised training to keep track of which layers should be trained by which loss function. 
        # Irrelevant for Non-specialised models
        self.loss_dict = {}

    def build(self, input_shapes):

        for k in range(self.n_hidden_layers):
            layer_l = Dense(
                units=self.n_units[k],
                activation=self.activation,
                name='hidden_l_' + str(k)
            )
            self.loss_dict['hidden_l_' + str(k)  + '/kernel:0'] = 0
            self.loss_dict['hidden_l_' + str(k)  + '/bias:0'] = 0
            self.layers.append(layer_l)

            layer_r = Dense(
                units=self.n_units[k],
                activation=self.activation,
                name='hidden_r_' + str(k)
            )
            self.loss_dict['hidden_r_' + str(k)  + '/kernel:0'] = 1
            self.loss_dict['hidden_r_' + str(k)  + '/bias:0'] = 1
            self.layers.append(layer_r)

        combine_layer = Wt_Add()

        for v in combine_layer.trainable_variables:
                self.loss_dict[v.name] = -1
        
        output_layer = Dense(
            units=self.plant.input_dim,
            activation='sigmoid',
            name='output_layer'
        )

        self.loss_dict['output_layer/kernel:0'] = -1
        self.loss_dict['output_layer/bias:0'] = -1

        self.layers.append(combine_layer)
        self.layers.append(output_layer)
        self.built = True

    def get_initial_state(self, inputs=None, batch_size: int = 1, dtype=tf.float32):
        states = self.get_base_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)
        #hidden_states = self.get_new_hidden_state((batch_size, dtype))
        #states.extend(hidden_states)
        return states

    def get_save_config(self):
        base_config = self.get_base_config()
        cfg = {
            'n_units': int(self.n_units[0]),
            'n_hidden_layers': self.n_hidden_layers,
            'activation': self.activation_name, **base_config
        }
        return cfg

    @tf.function
    def forward_pass(self, inputs, states):
        """Performs the forward pass computation.

        Args:
            inputs: `Tensor`, inputs to the first layer of the network.
            states: `List` of `tensor` arrays representing the states of each layer operating on a state (state-based
                layers).

        Returns:
            - A `tensor` array, the output of the last layer to use as the motor command, or excitation to the plant.
            - A `list` of the new hidden states of the GRU layers.
            - A `dictionary` of the new hidden states of the GRU layers.
        """
        new_hidden_states_dict = {}
        new_hidden_states = []
        xl = inputs
        xr = inputs

        if(self.lesion == "highleft" or self.lesion == "ccl"):
            xl = tf.multiply(xl,0)
        if(self.lesion == "highright" or self.lesion == "ccr"):
            xr = tf.multiply(xl,0)

        for k in range(self.n_hidden_layers):
            l_idx = k*2
            r_idx = (k*2) + 1
            if(k == 0 or self.lesion == "cc" or self.lesion == "ccl" or self.lesion == "ccr"):
                xl = self.layers[l_idx](xl)
                xr = self.layers[r_idx](xr)
            else:
                #left
                cc_r = tf.keras.layers.Reshape((np.shape(xr)[1], 1))(xr)
                cc_r = tf.keras.layers.MaxPooling1D(pool_size=2)(cc_r)
                cc_r = tf.keras.layers.ZeroPadding1D(padding=(np.shape(xl)[1]-np.shape(cc_r)[1],0))(cc_r)
                cc_r = tf.keras.layers.Reshape((np.shape(cc_r)[1],))(cc_r)
                #add to left
                xl_new = tf.keras.layers.Add()([xl, cc_r])

                #right
                cc_l = tf.keras.layers.Reshape((np.shape(xl)[1], 1))(xl)
                cc_l = tf.keras.layers.MaxPooling1D(pool_size=2)(cc_l)
                cc_l = tf.keras.layers.ZeroPadding1D(padding=(np.shape(xr)[1]-np.shape(cc_l)[1],0))(cc_l)
                cc_l = tf.keras.layers.Reshape((np.shape(cc_l)[1],))(cc_l)
                #add to right
                xr_new = tf.keras.layers.Add()([xr, cc_l])

                #normal layer operation
                xl = self.layers[l_idx](xl_new)
                xr = self.layers[r_idx](xr_new)
                
                

        
        if(self.lesion == "lowleft"):
            xl = tf.multiply(xl,0)
        if(self.lesion == "lowright"):
            xr = tf.multiply(xr,0)

        x = self.layers[-2](xl,xr)

        u = self.layers[-1](x)
        return u, new_hidden_states, new_hidden_states_dict

@tf.function
def recttanh(x):
    """A rectified hyperbolic tangent activation function."""
    x = tf.keras.activations.tanh(x)
    x = tf.where(tf.less_equal(x, tf.constant(0.)), tf.constant(0.), x)
    return x

class DistalTeacherDual(tf.keras.Model, ABC):
    # Specialised model teacher. Allows 2 loss function to be used through model
    """This is a custom ``tensorflow.keras.Model`` object, whose purpose is to enable saving
    ``motornet.plants`` object configuration when saving the model as well.

    In `Tensorflow`, ``tensorflow.keras.Model`` objects group layers into an object with training and inference
    features. See the Tensorflow documentation for more details on how to declare, compile and use use a
    ``tensorflow.keras.Model`` object.

    Conceptually, as this model class performs backward propagation through the plant (which can be considered a perfect
    forward model), this class essentially performs the training of the controller using a `distal teacher` algorithm,
    as defined in `[1]`.

    References:
        [1] `Jordan MI, Rumelhart DE. Forward Models: Supervised Learning with a Distal Teacher.
        Cognitive Science, 1992 Jul;16(3):307-354. doi: 10.1207/s15516709cog1603_1.`

    Args:
        inputs: A :class:`tensorflow.keras.layers.Input`` object or `list` of :class:`tensorflow.keras.layers.Input`
            objects that will serve as `tensor` placeholder input(s) to the model.
        outputs: The output(s) of the model. See `motornet` tutorial on how to build a model, and the introduction
            section of the Functional API example in the `Tensorflow` documentation for more information about this
            argument: https://www.tensorflow.org/guide/keras/functional#introduction.
        task: A :class:`motornet.tasks.Task` object class or subclass.
        name: `String`, the name of the model.
    """

    def __init__(self, inputs, outputs, task, dual_loss_weights, name='controller'):
        self.inputs = inputs
        self.outputs = outputs
        self.task = task
        super().__init__(inputs=inputs, outputs=outputs, name=name)

        # ensure each loss is tagged with the correct loss name, since the loss order is reshuffled in the parent
        # `tensorflow.keras.Model` class.
        flat_losses = tf.nest.flatten(task.losses)
        names = list(task.losses.keys())
        losses = list(task.losses.values())

        # all non-defined losses (=None) will share the output_name of the first model output with a non-defined loss,
        # but we will remove those after anyway.
        output_names = [names[losses.index(loss)] for loss in flat_losses]
        loss_names = [task.loss_names[name] for name in output_names]

        # now we remove the names for the non-defined losses (loss=None cases)
        for k, loss in enumerate(flat_losses):
            if loss is None:
                loss_names[k] = None

        # the name assigned to losses will be used as output instead of the actual state output names
        self.output_names = loss_names
        
        self.dual_loss_weights = dual_loss_weights
        
        self.dual_loss = []
        for w in dual_loss_weights:
            self.dual_loss =self.dual_loss +   [compile_utils.LossesContainer(
                    self.task.losses, w, output_names=None
                )]
        
        self.dual_adjust = []
        for l in range(len(self.dual_loss_weights)):
            adj = []
            for i in self.trainable_variables:
                if(self.task.network.loss_dict[i.name] == l):
                    adj = adj + [1]
                elif(self.task.network.loss_dict[i.name] == -1):
                    adj = adj + [0.5]
                else: 
                    adj = adj + [0]
            self.dual_adjust = self.dual_adjust + [copy.deepcopy(adj)]

    def train_step(self, data):
        """The logic for one training step. Compared to the default method, this overriding method allows for
        recomputation of targets online (during movement), in addition to essentially reproducing what the default
        method does.

        .. warning::
            Some features from the original :meth:`tensorflow.keras.Model.train_step` method are not implemented here.

            - Outputing metrics as a `list` instead of a `dictionary`, since `motornet` always uses dictionaries
            - The sample weighting functionality, since data in `motornet` is usually synthetic and not empirical,
              meaning there is usually no bias in sample representation.

        Args:
            data: A nested structure of `tensor` arrays.

        Returns:
            A `dictionary` containing values that will be passed to
            :meth:`tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the values of the `Model`'s metrics
            are returned. Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """

        x, y = data
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            y_pred = self(x, training=True)  # Forward pass

            # we can recompute y after the forward pass if it's required by the task
            if self.task.do_recompute_targets:
                y = self.task.recompute_targets(x, y, y_pred)

            # Compute the loss value (the compiled_loss method is configured in `self.compile()`)
            loss = []
            for l in self.dual_loss:
                loss = loss + [l(y, y_pred, regularization_losses=self.losses)]
            

        # Compute gradients
        trainable_vars = self.trainable_variables
        
        #calculate both loss gradients
        gradients = []
        gradients = gradients + [tape1.gradient(loss[0], trainable_vars)]
        gradients = gradients + [tape2.gradient(loss[1], trainable_vars)]

        #ensure each gradient is used in the right place
        for i in range(len(gradients)):
            gradients[i] = [tensor * integer for tensor, integer in zip(gradients[i], self.dual_adjust[i])]
        combined_grad = [g0 + g1 for g0, g1 in zip(gradients[0], gradients[1])]

        # Update weights
        self.optimizer.apply_gradients(zip(combined_grad, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def save_model(self, path, **kwargs):
        """Gets the model's configuration as a dictionary and then save it into a JSON file.

        Args:
            path: `String`, the absolute path to the JSON file that will be produced. The name of the JSON file itself
                should be included, without the extension. For instance, if we want to create a JSON file called
                `my_model_config.json` in `~/path/to/desired/directory`, we would call this method in the python console
                like so:

                .. code-block:: python

                    model.save_model("~/path/to/desired/directory/my_model_config")

            **kwargs: Not used here, this is for subclassing compatibility only.
        """
        cfg = {'Task': self.task.get_save_config()}
        cfg.update({'Network': self.task.network.get_save_config()})
        cfg.update({'Plant': self.task.network.plant.get_save_config()})
        if os.path.isfile(path + '.json'):
            raise ValueError('Configuration file already exists')
        else:
            with open(path + '.json', 'w+') as file:
                json.dump(cfg, file)

    def get_config(self):
        """Gets the model's configuration.

        Returns:
            A `dictionary` containing the model's configuration. This includes the task object passed at initialization.
        """

        cfg = super().get_config()
        cfg.update({'task': self.task, 'inputs': self.inputs, 'outputs': self.outputs})
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

class MotorNetModelDual(DistalTeacherDual):
    #Trains a specilaised model
    #Non specialised models use MotorNetModel class from the motornet toolbox instead
    """This is an alias name for the :class:`DistalTeacherDual` class for backward compatibility."""

    def __init__(self, inputs, outputs, task, dual_loss_weights, name='controller'):
        super().__init__(inputs=inputs, outputs=outputs, task=task, dual_loss_weights = dual_loss_weights, name=name)
