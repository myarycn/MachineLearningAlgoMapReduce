# ##################################################################### #
# File Desc: SVM MapReduce                                              #
# Author: Yixin Chen, University of Iowa.                               #
# Created: 11/21/2014                                                   #
# Python Version: 2.7                                                   #
#                                                                       #
# Note: this script implements SVM Algorithm using MapReduce.           #
# ##################################################################### #

import numpy as np
import sys
from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol
from mrjob.step import MRStep

# initiate weight
w = np.random.random_sample(3)
# initiate epsilon (this is to check if the distance between the previous weight and the current weight is smaller than eps)
eps = 0.001


def get_arg(args, target):
    value = 0
    for arg in args:
        if target in arg:
            arg_value = arg.split('=')
            if len(arg_value)==1:
                continue
            else:
                value = float(arg_value[1])
            break
    return value


# cosin similarity between two vectors
def cosine_distance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class MRsvm(MRJob):

    INTERNAL_PROTOCOL = PickleProtocol

    def __init__(self, *args, **kwargs):
        super(MRsvm, self).__init__(*args, **kwargs)
        self.data = open('svm_training_set.txt','rb').readlines()
        self.lamda = 0.2                # coeff
        self.step_size = 0.1            # step_size
        self.iterations = 1             # iteration number

    def configure_options(self):
        super(MRsvm, self).configure_options()
        self.add_passthrough_option(
            '--iterations', dest='iterations', default=2, type='int',
            help='T: number of iterations to run')

    def map(self, key, line): #needs exactly 2 arguments
        global w
        line = str(line).split('\t')
        x = np.array(line[:-1]+[1], dtype=np.float)
        label = int(line[-1])
        if label*np.dot(w.transpose(),x) >= 1:
            yield None, np.dot(-1*self.lamda, np.dot(x, label))
        else: # if y_i * W_T * x_i < 1, return 0.
            yield None, 0

    def reduce(self, _, value):
        global w
        w = np.subtract(w, np.dot(self.step_size, np.add(sum(value), w)))
        yield None, w

    def steps(self):
        return ([MRStep(mapper=self.map, reducer=self.reduce)])#*self.options.iterations)


if __name__ == '__main__':
    args = sys.argv[1:]
    iterations = int(get_arg(args, '--iterations'))
    count = 1
    print '---------- initial weight ----------'
    print w
    prev = np.random.random_sample(3)
    # prev = np.array([1,1,1])
    while count<=iterations:
        # print cosine_distance(w, prev)
        print 'Iteration {0}'.format(count)
        svm_job = MRsvm(args=args+['--iterations='+str(iterations)])
        with svm_job.make_runner() as svm_runner:
            svm_runner.run()
        if abs(cosine_distance(w, prev)-1.0)<=eps:
            print '----------- final weight ----------'
            print w
            break
        print '----------- new weight ----------'
        print w
        prev = w
        count += 1
