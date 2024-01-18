import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
auto_reduction = losses_utils.ReductionV2.AUTO

def _scale_activation(activation, max_iso_force):
    """Scale activation penalty by maximum muscle force of each muscle."""
    max_iso_force_n = max_iso_force / tf.reduce_mean(max_iso_force)
    return activation * tf.expand_dims(tf.expand_dims(max_iso_force_n, axis=0), axis=0)

def _muscle_loss(y_true, y_pred, max_iso_force):
    activation = y_pred
    return tf.reduce_mean(activation ** 2)

#def _weight_loss(y_true, y_pred, model, lambda1: float = 0.1):
#    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in model.trainable_weights])
#    return lambda1*l2_loss

def _position_loss2(y_true, y_pred):
    true_pos, _ = tf.split(y_true, 2, axis=-1)
    pred_pos, pred_vel = tf.split(y_pred, 2, axis=-1)
    # add a fixed penalty any time the arm hits the joint limits
    joint_limit_cost = tf.where(tf.equal(pred_vel[:, 1:, :], 0.), x=0., y=0.)
    return tf.reduce_mean(tf.square(10*(true_pos - pred_pos)))/10 + tf.reduce_mean(joint_limit_cost)


class MuscleLoss(LossFunctionWrapper):
    """Applies a L2 penalty to muscle activation. Must be applied to the `muscle state` output state.
    The L2 penalty is normalized by the maximum isometric force of each muscle.

    Args:
        max_iso_force: `Float` or `list`, the maximum isometric force of each muscle in the order they are declared in
            the :class:`motornet.plants.plants.Plant` object class or subclass.
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensorflow` documentation for more details.
    """

    def __init__(self, max_iso_force, name: str = 'muscle_activation', reduction=auto_reduction):
        super().__init__(_muscle_loss, name=name, reduction=reduction, max_iso_force=max_iso_force)
        self.max_iso_force = max_iso_force

#class WeightLoss(LossFunctionWrapper):
#    """Applies a L2 penalty to muscle activation. Must be applied to the `muscle state` output state.
#    The L2 penalty is normalized by the maximum isometric force of each muscle.
#
#    Args:
#        max_iso_force: `Float` or `list`, the maximum isometric force of each muscle in the order they are declared in
#            the :class:`motornet.plants.plants.Plant` object class or subclass.
#        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
#            losses during training.
#        reduction: The reduction method used. The default value is
#           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
#           See the `Tensorflow` documentation for more details.
#    """
#
#    def __init__(self, model, name: str = 'weight_regu', reduction=auto_reduction):
#        super().__init__(_weight_loss, name=name, reduction=reduction, model=model)

class PositionLoss2(LossFunctionWrapper):
    """Applies a L1 penalty to positional error between the model's output positional state ``x`` and a user-fed
    label position ``y``:

    .. code-block:: python

        xp, _ = np.split(x, 2, axis=-1)  # remove velocity from the positional state
        yp, _ = np.split(y, 2, axis=-1)
        loss = np.reduce_mean(np.abs(xp - yp))

    .. note::
        The positional error does not include velocity, hence the use of ``np.split`` to extract position from the
        state array.

    Args:
        name: `String`, the name (label) to give to the compounded loss object. This is used to print, plot, and save
            losses during training.
        reduction: The reduction method used. The default value is
           ``tensorflow.python.keras.utils.losses_utils.ReductionV2.AUTO``.
           See the `Tensoflow` documentation for more details.
    """

    def __init__(self, name: str = 'position2', reduction=auto_reduction):
        super().__init__(_position_loss2, name=name, reduction=reduction)
