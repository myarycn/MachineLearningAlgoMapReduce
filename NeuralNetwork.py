# ##################################################################### #
# File Desc: Neural Network MapReduce                                   #
# Author: Yixin Chen, University of Iowa.                               #
# Created: 10/15/2014                                                   #
# Python Version: 2.7                                                   #
#                                                                       #
# Note: it implements ANN Algorithm (Backpropagation, three layers)     #
#       using MapReduce. I use a three layer network with one output    #
#       neurons to classify the data into two categories, each mapper   #
#       propagates its set of data through the network. For each        #
#       training example, the error is back propagated to calculate     #
#       the partial gradient for each of the weights in the network.    #
#       The reducer then sums the partial gradient from each mapper     #
#       and does a batch gradient descent to update the weights of      #
#       the network.                                                    #
#       Global variable is only for testing, should be stored in file   #
#       or passing it to job config if running on AWS.                  #
#                                                                       #
# Steps:                                                                #
# 1. define the network structure                                       #
#               [number_of_neurons_in_input_layer,                      #
#                number_of_neurons_in_hidden_layer,                     #
#                number_of_neurons_in_output_layer,],                   #
#    if there're n features, number_of_neurons_in_input_layer = n       #
# 2. initiate biases of all the neurons and weight of all edges.        #
# 3. Each mapper applies Backpropagation to calculate delta_biases      #
#    and delta_weights.                                                 #
# 4. Each reducer sums the gradients (delta_biases/delta_weights) up,   #
#    and get the average.                                               #
# 5. yield new updated biase and weights.                               #
#                                                                       #
#                                                                       #
# Formulas used:                                                        #
#                     hidder_layer        output_layer                  #
#                                                                       #
#          Zs = { [net_a, net_b, ...],      [net_z]    }  # net input   #
# activations = { [O_a,    O_b,  ...],      [O_z]      }  # output      #
#      deltas = { [d_a,    d_b, ...],       [d_z]      }                #
# d_i =  O_i*(1-O_i)*sum(d_i*weight(i,successors)) for hidden laryer    #
# d_i =  (t_i-O_i)*O_i*(1-O_i)                     for output laryer    #
#    delta_weight(i, j) = learning_rate * d_j * O_i                     #
#      delta_bias(i, j) = learning_rate * d_j                           #
#  new_weight = weight + (learning_rate/sum(occurrence))*delta_weight   #
#  new_bias   = bias + (learning_rate/sum(occurrence))*delta_bias       #
#                                                                       #
#                                                                       #
# Input file format (numerical):                                        #
#        value11,   value21,   ..., label1                              #
#        value12,   value22,   ..., label2                              #
#        value13,   value23,   ..., label3                              #
#          ...        ...      ...                                      #
#      { note: each value should be a float number between 0 and 1 }    #
#                                                                       #
# Running cmd args:                                                     #
#        ----learning_rate=? --iteration=?                              #
#        { note: 0 < learning rate < 1 }                                #
#                                                                       #
# Output format:                                                        #
#        Iteration 0:                                                   #
#           list of updated biases, list of updated weights             #
#        Iteration 1:                                                   #
#           list of updated biases, list of updated weights             #
#        Iteration 2:                                                   #
#                   ...                                                 #
#        The last iteration would be the final biases and weights.      #
#                                                                       #
# How to Classifying testing set:                                       #
#        Apply the algorithm on each instance of the testing set,       #
#        calculate y = weights * instance + biases,                     #
#        if y > 0, fire; else, None.                                    #
# ##################################################################### #


from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol
import numpy as np
import sys

network = [13,3,1]
biases = [np.random.randn(1, y) for y in network[1:]]
weights = [np.random.randn(y, x) for x, y in zip(network[:-1], network[1:])]


# BackPropagation
def back_prop(instance, label, biases, weights, learning_rate):
        # forward pass
        delta_w = [np.zeros(w.shape) for w in weights]
        zs = []
        activations = []
        instance_cp = np.array(instance[:])
        for b, w in zip(biases, weights):
            z = np.array(np.dot(w, instance_cp.transpose()))
            # net_a, net_b, net_c, ...
            zs.append(z)
            z = np.add(z,b)
            instance_cp = b
            # O_a, O_b, O_c, ...
            activation = sigmoid_vec(z)
            activations.append(activation)
        # backward pass
        deltas = derivative(activations, label , 'hidden', weights)
        delta_bias = derivative_alpha(activations, label , 'hidden', weights, learning_rate)

        # update delta weight, delta_w[-1] represents weight of neurons in the output layer,
        # delta_w[-2] represents weight of neurons in the hidden layer.
        delta_w[-1] = np.dot(np.dot(deltas[-1], activations[-2]), learning_rate)
        delta_w[-2] = np.dot(np.dot(deltas[-2].transpose(), [instance]), learning_rate)
        return (delta_bias, delta_w)


def get_bias_weight(job, runner):
    return biases, weights

# sum of two numpy arrays
def plus(delta_value, orig_value, type):
    b = []
    for delta, orig in zip(delta_value, orig_value):
        if type == 'bias':
            orig_b = np.array(orig).transpose()
        else:
            orig_b = np.array(orig)
        delta = np.array(delta)
        t = []
        for x, y in zip(delta, orig):
            t.append(x+y)
        t = np.asarray(t)
        b.append(t)
    return b

# readin arguments from command line
def get_arg(args, target):
    value = 0
    for arg in args:
        if target in arg:
            value = float(arg.split('=')[1])
            break
    return value

# The sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
sigmoid_vec = np.vectorize(sigmoid)


# calculate deltas
def derivative(activations, output_label, type_of_layer, weights):
        if type_of_layer == 'output':
            return (output_label - activations[-1]) * activations[-1] * (1 - activations[-1])
        elif type_of_layer == 'hidden':
            deltas = []
            for o, w in zip(activations[-2], weights[-1]):
                delta_output_node = derivative(activations, output_label, 'output', weights)
                delta = o * (1-o) * delta_output_node * w
                deltas.append(delta)
            deltas.append(delta_output_node)
            return deltas

# calculate deltas
def derivative_alpha(activations, output_label, type_of_layer, weights, learning_rate):
        if type_of_layer == 'output':
            return (output_label - activations[-1]) * activations[-1] * (1 - activations[-1])
        elif type_of_layer == 'hidden':
            deltas = []
            for o, w in zip(activations[-2], weights[-1]):
                delta_output_node = derivative_alpha(activations, output_label, 'output', weights, learning_rate)
                delta = o * (1-o) * delta_output_node * w
                deltas.append(delta*learning_rate)
            deltas.append(delta_output_node*learning_rate)
            return deltas



class ANN(MRJob):

    INTERNAL_PROTOCOL = PickleProtocol
    #learning_rate = 0.1

    def configure_options(self):
        super(ANN, self).configure_options()
        self.add_passthrough_option('--iteration', dest='iteration', default=1, type='int', help='number of iterations')
        self.add_passthrough_option('--learning_rate', dest='learning_rate', default=0.1, type='float', help='learning rate')

    def mapper(self, _, line):
        global biases, weights
        line = map(float, line.split(','))
        # x -- instance, y -- label
        x = list(line[:-1])
        y = tuple(line[-1:])
        alpha = self.options.learning_rate # learning rate
        (delta_bias, delta_w) = back_prop(x, y, biases, weights, alpha)
        yield None, (delta_bias, delta_w, 1)


    def reducer(self, key, values):
        global biases, weights
        value = values.next()
        # get delta_bias, delta_weight, number
        value_bias = np.array(value[0])
        value_weight = np.array(value[1])
        value_number = np.array(value[2])
        for v in values:
            # get delta_bias, delta_weight, number of instances
            v_b = np.array(v[0])
            v_w = np.array(v[1])
            v_n = np.array(v[2])
            value_bias = np.array(np.add(value_bias, v_b))
            value_weight = np.array(np.add(value_weight, v_w))
            value_number = np.add(value_number, v_n)
        value_bias = np.array(np.multiply(value_bias, np.array(learning_rate/value_number)))
        # get average delta weight
        value_weight = np.multiply(value_weight, np.array(learning_rate/value_number))
        biases = plus(value_bias, biases, 'bias')
        weights = plus(value_weight, weights, 'weight')
        yield biases, weights



    def steps(self):
        return [
            self.mr(mapper=self.mapper,
                    reducer=self.reducer)
        ]

if __name__ == '__main__':
    args = sys.argv[1:]
    iterations = int(get_arg(args, '--iteration'))
    learning_rate = get_arg(args, '--learning_rate')
    count = 0
    while count<iterations:
        print 'Iteration {0}'.format(count)
        ANN_job = ANN(args=args+['--learning_rate='+str(learning_rate)]+['--iteration='+str(iterations)])
        with ANN_job.make_runner() as ANN_job_runner:
            ANN_job_runner.run()
        new_bias, new_weight = get_bias_weight(ANN_job, ANN_job_runner)
        print 'update biases -->'
        print new_bias
        print 'update weights -->'
        print new_weight
        count += 1