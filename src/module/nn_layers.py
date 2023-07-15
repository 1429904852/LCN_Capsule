import tensorflow as tf


def bi_dynamic_rnn(inputs, length, hidden_size, scope_name):
    (outputs_fw_q, outputs_bw_q), state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(hidden_size),
        cell_bw=tf.contrib.rnn.LSTMCell(hidden_size),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    hidden = tf.concat([outputs_fw_q, outputs_bw_q], -1)
    return hidden


def bi_dynamic_gru(inputs, length, hidden_size, scope_name):
    (outputs_fw_q, outputs_bw_q), state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.nn.rnn_cell.GRUCell(hidden_size),
        cell_bw=tf.nn.rnn_cell.GRUCell(hidden_size),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    hidden = tf.concat([outputs_fw_q, outputs_bw_q], -1)
    return hidden


def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    return inputs


def last_mean_with_len(outputs, n_hidden, length, max_len):
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * max_len + (length - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * 2n_hidden+1
    return outputs
