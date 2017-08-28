import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf


OUTPUT_DIR = 'output/'
STYLE_IMAGE = 'impression-sunrise.jpg'
IMAGE_WIDTH = 976
IMAGE_HEIGHT = 800
COLOR_CHANNELS = 3

ITERATIONS = 100

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# The mean to subtract from the input to the VGG model. This is the mean that
# when the VGG was used to train. Minor changes to this will make a lot of
# difference to the performance of model.
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def generate_noise_image():
    noise_image = np.random.uniform(
            -20, 20,
            (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    return noise_image

def load_image(path):
    image = scipy.misc.imread(path)
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    # Input to the VGG model expects the mean to be subtracted
    print(image.shape)
    image = image - MEAN_VALUES
    return image

def save_image(path, image):
    # Output should add back the mean.
    image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def load_vgg_model(path):

    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        W = vgg_layers[0][layer][0][0][0][0][0]
        b = vgg_layers[0][layer][0][0][0][0][1]
        layer_name = vgg_layers[0][layer][0][0][-2]
        assert layer_name == expected_layer_name
        return W, b

    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(
            prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d_relu(prev_layer, layer, layer_name):
        return _relu(_conv2d(prev_layer, layer, layer_name))


    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    return graph

def style_loss_func(sess, model):

    def _gram_matrix(F, N, M):
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        l = (1 / (4 * a.shape[3]**2 * a.shape[1] * a.shape[2]**2)) * tf.reduce_sum(tf.pow(_gram_matrix(x, a.shape[3], a.shape[1] * a.shape[2]) - _gram_matrix(a, a.shape[3], a.shape[1] * a.shape[2]), 2))
        return l

#paper defaults: 0.5  1  1.5   3  4
    layers = [
        ('conv1_1', 0),
        ('conv2_1', 0.5),
        ('conv3_1', 1.25),
        ('conv4_1', 2.5),
        ('conv5_1', 5),
    ]
    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
    W = [w for _, w in layers]
    loss = [W[l] * E[l] for l in range(len(layers))]
    return loss


if __name__ == '__main__':
    with tf.Session() as sess:

        style_source = load_image(STYLE_IMAGE)
        model = load_vgg_model(VGG_MODEL)
        input_image = generate_noise_image()

        sess.run(tf.initialize_all_variables())
        sess.run(model['input'].assign(style_image))
        style_loss = style_loss_func(sess, model)

        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(style_loss)

        sess.run(tf.initialize_all_variables())
        sess.run(model['input'].assign(input_image))

        for i in range(ITERATIONS):
            sess.run(train_step)
        style_image = sess.run(model['input'])

        save_image('style.png', style_image)
