from math import sqrt
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.linalg
import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
import geomstats.backend as gs # import the Geomstats tensorflow 2.2 backend
tf.compat.v1.disable_eager_execution()

FLOAT_TYPE = gs.float32
EPSILON = 1e-8


# Layers for 3D rotation-equivariant network.

def get_eijk():
    """
    Constant Levi-Civita tensor
    Returns:
        tf.Tensor of shape [3, 3, 3]
    """
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return tf.constant(eijk_, dtype=FLOAT_TYPE)


def norm_with_epsilon(input_tensor, axis=None, keep_dims=False):
    """
    Regularized norm
    Args:
        input_tensor: tf.Tensor
    Returns:
        tf.Tensor normed over axis
    """
    return gs.sqrt(gs.maximum(tf.reduce_sum(tf.square(input_tensor), axis=axis, keep_dims=keep_dims), EPSILON))


def ssp(x):
    """
    Shifted soft plus nonlinearity.
    Args:
        x: tf.Tensor
    Returns:
        tf.Tensor of same shape as x 
   """
    return gs.log(0.5 * gs.exp(x) + 0.5)


def rotation_equivariant_nonlinearity(x, nonlin=ssp, biases_initializer=None):
    """
    Rotation equivariant nonlinearity.
    The -1 axis is assumed to be M index (of which there are 2 L + 1 for given L).
    Args:
        x: tf.Tensor with channels as -2 axis and M as -1 axis.
    Returns:
        tf.Tensor of same shape as x with 3d rotation-equivariant nonlinearity applied.
    """
    if biases_initializer is None:
        biases_initializer = tf.constant_initializer(0.)
    shape = x.get_shape().as_list()
    channels = shape[-2]
    representation_index = shape[-1]

    biases = tf.get_variable('biases',
                             [channels],
                             dtype=FLOAT_TYPE,
                             initializer=biases_initializer)

    if representation_index == 1:
        return nonlin(x)
    else:
        norm = norm_with_epsilon(x, axis=-1)
        nonlin_out = nonlin(tf.nn.bias_add(norm, biases))
        factor = gs.divide(nonlin_out, norm)
        # Expand dims for representation index.
        return tf.multiply(x, gs.expand_dims(factor, axis=-1))
    


def difference_matrix(geometry):
    """
    Get relative vector matrix for array of shape [N, 3].
    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]
    Returns:
        Relative vector matrix with shape [N, N, 3]
    """
    # [N, 1, 3]
    ri = gs.expand_dims(geometry, axis=1)
    # [1, N, 3]
    rj = gs.expand_dims(geometry, axis=0)
    # [N, N, 3]
    rij = ri - rj
    return rij


def distance_matrix(geometry):
    """
    Get relative distance matrix for array of shape [N, 3].
    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]
    Returns:
        Relative distance matrix with shape [N, N]
    """
    # [N, N, 3]
    rij = difference_matrix(geometry)
    # [N, N]
    dij = norm_with_epsilon(rij, axis=-1)
    return dij


def random_rotation_matrix(numpy_random_state):
    """
    Generates a random 3D rotation matrix from axis and angle.
    Args:
        numpy_random_state: numpy random state object
    Returns:
        Random rotation matrix.
    """
    rng = numpy_random_state
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + EPSILON
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))


def R(inputs, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "radial_function", values=[inputs]):
        if weights_initializer is None:
            weights_initializer = tf.keras.initializers.glorot_normal()
        if biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)
        input_dim = inputs.get_shape()[-1]
        if hidden_dim is None:
            hidden_dim = input_dim

        w1 = tf.get_variable('weights1', [hidden_dim, input_dim], dtype=FLOAT_TYPE,
                             initializer=weights_initializer)
        b1 = tf.get_variable('biases1', [hidden_dim], dtype=FLOAT_TYPE, initializer=biases_initializer)

        w2 = tf.get_variable('weights2', [output_dim, hidden_dim], dtype=FLOAT_TYPE,
                             initializer=weights_initializer)
        b2 = tf.get_variable('biases2', [output_dim], dtype=FLOAT_TYPE, initializer=biases_initializer)

        hidden_layer = nonlin(b1 + tf.tensordot(inputs, w1, [[2], [1]]))
        radial = b2 + tf.tensordot(hidden_layer, w2, [[2], [1]])

        # [N, N, output_dim]
        return radial


def unit_vectors(v, axis=-1):
    return v / norm_with_epsilon(v, axis=axis, keep_dims=True)


def Y_2(rij):
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = tf.maximum(tf.reduce_sum(tf.square(rij), axis=-1), EPSILON)
    # return : [N, N, 5]
    output = tf.stack([x * y / r2,
                       y * z / r2,
                       (-tf.square(x) - tf.square(y) + 2. * tf.square(z)) / (2 * sqrt(3) * r2),
                       z * x / r2,
                       (tf.square(x) - tf.square(y)) / (2. * r2)],
                      axis=-1)
    return output


def F_0(inputs, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    # [N, N, output_dim, 1]
    with tf.variable_scope(None, "F_0", values=[inputs]):
        return tf.expand_dims(
            R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
              weights_initializer=weights_initializer, biases_initializer=biases_initializer),
            axis=-1)


def F_1(inputs, rij, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "F_1", values=[inputs]):
        # [N, N, output_dim]
        radial = R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                   weights_initializer=weights_initializer, biases_initializer=biases_initializer)
        # Mask out for dij = 0
        dij = tf.norm(rij, axis=-1)
        condition = tf.tile(tf.expand_dims(dij < EPSILON, axis=-1), [1, 1, output_dim])
        masked_radial = tf.where(condition, tf.zeros_like(radial), radial)
        # [N, N, output_dim, 3]
        return tf.expand_dims(unit_vectors(rij), axis=-2) * tf.expand_dims(masked_radial, axis=-1)


def F_2(inputs, rij, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "F_2", values=[inputs]):
        # [N, N, output_dim]
        radial = R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                   weights_initializer=weights_initializer, biases_initializer=biases_initializer)
        # Mask out for dij = 0
        dij = tf.norm(rij, axis=-1)
        condition = tf.tile(tf.expand_dims(dij < EPSILON, axis=-1), [1, 1, output_dim])
        masked_radial = tf.where(condition, tf.zeros_like(radial), radial)
        # [N, N, output_dim, 5]
        return tf.expand_dims(Y_2(rij), axis=-2) * tf.expand_dims(masked_radial, axis=-1)


def filter_0(layer_input,
             rbf_inputs,
             nonlin=tf.nn.relu,
             hidden_dim=None,
             output_dim=1,
             weights_initializer=None,
             biases_initializer=None):
    with tf.variable_scope(None, "F0_to_L", values=[layer_input]):
        # [N, N, output_dim, 1]
        F_0_out = F_0(rbf_inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                      weights_initializer=weights_initializer, biases_initializer=biases_initializer)
        # [N, output_dim]
        input_dim = layer_input.get_shape().as_list()[-1]
        # Expand filter axis "j"
        cg = tf.expand_dims(tf.eye(input_dim), axis=-2)
        # L x 0 -> L
        return tf.einsum('ijk,abfj,bfk->afi', cg, F_0_out, layer_input)


def filter_1_output_0(layer_input,
                      rbf_inputs,
                      rij,
                      nonlin=tf.nn.relu,
                      hidden_dim=None,
                      output_dim=1,
                      weights_initializer=None,
                      biases_initializer=None):
    with tf.variable_scope(None, "F1_to_0", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_1_out = F_1(rbf_inputs,
                      rij,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)
        # [N, output_dim, 3]
        if layer_input.get_shape().as_list()[-1] == 1:
            raise ValueError("0 x 1 cannot yield 0")
        elif layer_input.get_shape().as_list()[-1] == 3:
            # 1 x 1 -> 0
            cg = tf.expand_dims(tf.eye(3), axis=0)
            return tf.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")


def filter_1_output_1(layer_input,
                      rbf_inputs,
                      rij,
                      nonlin=tf.nn.relu,
                      hidden_dim=None,
                      output_dim=1,
                      weights_initializer=None,
                      biases_initializer=None):
    with tf.variable_scope(None, "F1_to_1", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_1_out = F_1(rbf_inputs,
                      rij,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)
        # [N, output_dim, 3]
        if layer_input.get_shape().as_list()[-1] == 1:
            # 0 x 1 -> 1
            cg = tf.expand_dims(tf.eye(3), axis=-1)
            return tf.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
        elif layer_input.get_shape().as_list()[-1] == 3:
            # 1 x 1 -> 1
            return tf.einsum('ijk,abfj,bfk->afi', get_eijk(), F_1_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")


def filter_2_output_2(layer_input,
                      rbf_inputs,
                      rij,
                      nonlin=tf.nn.relu,
                      hidden_dim=None,
                      output_dim=1,
                      weights_initializer=None,
                      biases_initializer=None):
    with tf.variable_scope(None, "F2_to_2", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_2_out = F_2(rbf_inputs,
                      rij,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)
        # [N, output_dim, 5]
        if layer_input.get_shape().as_list()[-1] == 1:
            # 0 x 2 -> 2
            cg = tf.expand_dims(tf.eye(5), axis=-1)
            return tf.einsum('ijk,abfj,bfk->afi', cg, F_2_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")


def self_interaction_layer_without_biases(inputs, output_dim, weights_initializer=None, biases_initializer=None):
    # input has shape [N, C, 2L+1]
    # input_dim is number of channels
    if weights_initializer is None:
        weights_initializer = tf.orthogonal_initializer()
    if biases_initializer is None:
        biases_initializer = tf.constant_initializer(0.)

    with tf.variable_scope(None, "self_interaction_layer", values=[inputs]):
        input_dim = inputs.get_shape().as_list()[-2]
        w_si = tf.get_variable('weights', [output_dim, input_dim], dtype=FLOAT_TYPE,
                               initializer=weights_initializer)
        # [N, output_dim, 2l+1]
        return tf.transpose(tf.einsum('afi,gf->aig', inputs, w_si), perm=[0, 2, 1])


def self_interaction_layer_with_biases(inputs, output_dim, weights_initializer=None, biases_initializer=None):
    # input has shape [N, C, 2L+1]
    # input_dim is number of channels
    if weights_initializer is None:
        weights_initializer = tf.orthogonal_initializer()
    if biases_initializer is None:
        biases_initializer = tf.constant_initializer(0.)

    with tf.variable_scope(None, "self_interaction_layer", values=[inputs]):
        input_dim = inputs.get_shape().as_list()[-2]
        w_si = tf.get_variable('weights', [output_dim, input_dim], dtype=FLOAT_TYPE,
                               initializer=weights_initializer)
        b_si = tf.get_variable('biases', [output_dim], initializer=biases_initializer,
                               dtype=FLOAT_TYPE)

        # [N, output_dim, 2l+1]
        return tf.transpose(tf.einsum('afi,gf->aig', inputs, w_si) + b_si, perm=[0, 2, 1])


def convolution(input_tensor_list, rbf, unit_vectors, weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "convolution", values=[input_tensor_list]):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
                for i, tensor in enumerate(input_tensor_list[key]):
                    output_dim = tensor.get_shape().as_list()[-2]
                    with tf.variable_scope(None, 'tensor_' + str(i), values=[tensor]):
                        tensor = tf.identity(tensor, name="in_tensor")
                        if True:
                            # L x 0 -> L
                            tensor_out = filter_0(tensor,
                                                  rbf,
                                                  output_dim=output_dim,
                                                  weights_initializer=weights_initializer,
                                                  biases_initializer=biases_initializer)
                            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                            tensor_out = tf.identity(tensor_out, name="F0_to_L_out_tensor")
                            output_tensor_list[m].append(tensor_out)
                        if key is 1:
                            # L x 1 -> 0
                            tensor_out = filter_1_output_0(tensor,
                                                           rbf,
                                                           unit_vectors,
                                                           output_dim=output_dim,
                                                           weights_initializer=weights_initializer,
                                                           biases_initializer=biases_initializer)
                            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                            tensor_out = tf.identity(tensor_out, name="F1_to_0_out_tensor")
                            output_tensor_list[m].append(tensor_out)
                        if key is 0 or key is 1:
                            # L x 1 -> 1
                            tensor_out = filter_1_output_1(tensor,
                                                           rbf,
                                                           unit_vectors,
                                                           output_dim=output_dim,
                                                           weights_initializer=weights_initializer,
                                                           biases_initializer=biases_initializer)
                            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                            tensor_out = tf.identity(tensor_out, name="F1_to_1_out_tensor")
                            output_tensor_list[m].append(tensor_out)
        return output_tensor_list


def self_interaction(input_tensor_list, output_dim, weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "self_interaction", values=[input_tensor_list]):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
                for i, tensor in enumerate(input_tensor_list[key]):
                    with tf.variable_scope(None, 'tensor_' + str(i), values=[tensor]):
                        if key == 0:
                            tensor_out = self_interaction_layer_with_biases(tensor,
                                                                            output_dim,
                                                                            weights_initializer=weights_initializer,
                                                                            biases_initializer=biases_initializer)
                        else:
                            tensor_out = self_interaction_layer_without_biases(tensor,
                                                                               output_dim,
                                                                               weights_initializer=weights_initializer,
                                                                               biases_initializer=biases_initializer)
                        m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                        output_tensor_list[m].append(tensor_out)
        return output_tensor_list


def nonlinearity(input_tensor_list, nonlin=tf.nn.elu, biases_initializer=None):
    with tf.variable_scope(None, "nonlinearity", values=[input_tensor_list]):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
                for i, tensor in enumerate(input_tensor_list[key]):
                    with tf.variable_scope(None, 'tensor_' + str(i), values=[tensor]):
                        if key == 0:
                            tensor_out = rotation_equivariant_nonlinearity(tensor,
                                                                                 nonlin=nonlin,
                                                                                 biases_initializer=biases_initializer)
                        else:
                            tensor_out = rotation_equivariant_nonlinearity(tensor,
                                                                                 nonlin=nonlin,
                                                                                 biases_initializer=biases_initializer)
                        m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                        output_tensor_list[m].append(tensor_out)
        return output_tensor_list


def concatenation(input_tensor_list):
    output_tensor_list = {0: [], 1: []}
    for key in input_tensor_list:
        with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
            # Concatenate along channel axis
            # [N, channels, M]
            output_tensor_list[key].append(tf.concat(input_tensor_list[key], axis=-2))
    return output_tensor_list
