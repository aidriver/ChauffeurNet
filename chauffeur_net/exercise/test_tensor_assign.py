# -*-coding:utf-8-*-
import tensorflow as tf

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def tensor_expand(tensor_Input, Num):
    '''
    张量自我复制扩展，将Num个tensor_Input串联起来，生成新的张量，
    新的张量的shape=[tensor_Input.shape,Num]
    :param tensor_Input:
    :param Num:
    :return:
    '''
    tensor_Input = tf.expand_dims(tensor_Input, axis=0)
    tensor_Output = tensor_Input
    for i in range(Num - 1):
        tensor_Output = tf.concat([tensor_Output, tensor_Input], axis=0)
    return tensor_Output


def get_one_hot_matrix(height, width, position):
    '''
    生成一个 one_hot矩阵，shape=【height*width】，在position处的元素为1，其余元素为0
    :param height:
    :param width:
    :param position: 格式为【h_Index,w_Index】,h_Index,w_Index为int格式
    :return:
    '''
    col_length = height
    row_length = width
    col_one_position = position[0]
    row_one_position = position[1]
    rows_num = height
    cols_num = width

    single_row_one_hot = tf.one_hot(row_one_position, row_length, dtype=tf.float32)
    single_col_one_hot = tf.one_hot(col_one_position, col_length, dtype=tf.float32)
    print(single_row_one_hot.eval())
    print(single_col_one_hot.eval())
    one_hot_rows = tensor_expand(single_row_one_hot, rows_num)
    one_hot_cols = tensor_expand(single_col_one_hot, cols_num)
    print(one_hot_rows.eval())
    print(one_hot_cols.eval())
    one_hot_cols = tf.transpose(one_hot_cols)
    print('trans:', one_hot_cols.eval())

    one_hot_matrx = one_hot_rows * one_hot_cols
    print(one_hot_matrx.eval())
    return one_hot_matrx


def tensor_assign_2D(tensor_input, position, value):
    '''
    给 2D tensor的特定位置元素赋值
    :param tensor_input: 输入的2D tensor，目前只支持2D
    :param position: 被赋值的张量元素的坐标位置，=【h_index,w_index】
    :param value:
    :return:
    '''
    shape = tensor_input.get_shape().as_list()
    height = shape[0]
    width = shape[1]
    h_index = position[0]
    w_index = position[1]
    one_hot_matrix = get_one_hot_matrix(height, width, position)
    #print(one_hot_matrix.eval())
    #print(tensor_input.eval())
    #print(tensor_input[h_index, w_index].eval())
    #print(value)
    #print((one_hot_matrix * value).eval())
    new_tensor = tensor_input - tensor_input[h_index, w_index] * one_hot_matrix + one_hot_matrix * value

    return new_tensor

import tensorflow as tf
import numpy as np

batch_size = 10
max_len = 2
dim = 3

'''
def integrate(integrate_input):
    Xy = integrate_input[0]
    Yx = integrate_input[1]
    #print(Xy.eval())
    #print(Yx.eval())
    #assert dimensions_equal(Xy.shape, (max_len, dim,))
    #assert dimensions_equal(Xy.shape, (max_len, dim,))
    outputs = tf.concat([Xy, Yx], 1)
    #print(outputs.eval())
    return outputs
'''


def integrate(Xy,Yx):
    #print(Xy.eval())
    #print(Yx.eval())
    #assert dimensions_equal(Xy.shape, (max_len, dim,))
    #assert dimensions_equal(Xy.shape, (max_len, dim,))
    #outputs = tf.concat([Xy, Yx], 1)
    #print(outputs.eval())
    #outputs = Xy * np.transpose(Yx)
    #outputs = Xy * tf.transpose(Yx)
    #outputs = Xy * Yx

    outputs = Xy + Yx
    return outputs

'''
def integrate(integrate_input):
    outputs = integrate_input * 2.0
    return outputs
'''


def my_fn(x, y):
    return x * y

def max_sentence_similarity(sentence_input, similarity_matrix):
    """
    Parameters
    ----------
    sentence_input: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim).
    similarity_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words, num_sentence_words).
    """
    # Shape: (batch_size, passage_len)
    def single_instance(inputs):
        single_sentence = inputs[0]
        argmax_index = inputs[1]
        # Shape: (num_sentence_words, rnn_hidden_dim)
        return tf.gather(single_sentence, argmax_index)

    question_index = tf.arg_max(similarity_matrix, 2)
    elems = (sentence_input, question_index)
    # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
    return tf.map_fn(single_instance, elems, dtype="float")

if __name__ == "__main__":
    '''
    a = np.array([[1, 2, 3], [2, 4, 1], [5, 1, 7]])
    b = np.array([[1, -1, -1], [1, 1, 1], [-1, 1, -1]])
    #print(a)
    print(a.shape)
    #print(b)
    print(b.shape)
    elems = (a, b)

    B = tf.map_fn(lambda x: my_fn(x[0], x[1]), elems, dtype=tf.int64)
    print(B.eval())
    exit(0)
    '''

    '''
    elems = np.array([1, 2, 3, 4, 5, 6])
    squares = tf.map_fn(lambda x: x * x, elems)
    # squares == [1, 4, 9, 16, 25, 36]

    elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
    alternate = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
    print(alternate)
    print(alternate.eval())
    # alternate == [-1, 2, -3]

    elems = np.array([1, 2, 3])
    alternates = tf.map_fn(lambda x: (x, -x), elems, dtype=(tf.int64, tf.int64))
    print(alternates)
    for alternate in alternates:
        print(alternate.eval())
    # alternates[0] == [1, 2, 3]
    # alternates[1] == [-1, -2, -3]
    exit(0)
    '''
    ##test
    '''
    tensor_input = tf.constant([i for i in range(20)], tf.float32)
    tensor_input = tf.reshape(tensor_input, [4, 5])
    print(tensor_input.eval())

    new_tensor = tensor_assign_2D(tensor_input, [2, 3], 100)
    print(new_tensor.eval())
    '''
    # 若inputs1,inputs2的shape为[batch_size,max_len,dim]
    # map_fn将inputs1、inputs2按第0维展开
    # 每次执行integrate时，Xy的shape为[max_len, dim]
    inputs1 = np.ones((batch_size, max_len, dim), dtype='float32')
    #inputs1 = np.ones((batch_size, max_len), dtype='float32')
    inputs1 = 2.0 * inputs1
    print(inputs1.shape)
    #inputs1 = tf.constant(2.0 * inputs1)
    inputs2 = np.ones((batch_size, max_len, dim), dtype='float32')
    #inputs2 = np.ones((batch_size, max_len), dtype='float32')
    inputs2 = 3.0 * inputs2
    print(inputs2.shape)
    #inputs2 = tf.constant(3.0 * inputs2)
    #print(inputs1.eval())
    #print(inputs2.eval())
    #outputs = tf.map_fn(integrate, (inputs1, inputs2), dtype=(tf.float32, tf.float32))
    '''
    inputs = np.concatenate([inputs1, inputs2], axis=-1)
    inputs = tf.constant(inputs1)
    outputs = tf.map_fn(integrate, inputs, dtype=tf.float32)
    print(outputs.eval())
    '''
    #outputs = tf.map_fn(integrate, (inputs1, inputs2), dtype=(tf.float32, tf.float32))
    outputs = tf.map_fn(lambda x: integrate(x[0], x[1]), (inputs1, inputs2), dtype=(tf.float32, tf.float32))# dtype=tf.float32, dtype=(tf.float32, tf.float32)
    print(outputs.eval())
