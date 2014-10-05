# ##################################################################### #
# File Desc: KMeans MapReduce                                           #
# Author: Yixin Chen, University of Iowa.                               #
# Created: 10/05/2014                                                   #
# Python Version: 2.7                                                   #
# Note: this script implements KMeans Algorithm using MapReduce.        #
#                                                                       #
# Steps: randomly choose k points as initial centroids from the         #
#        training set, for each input point x, assign to its            #
#        belonging cluster by computing the shortest euclidean          #
#        distance between point x and the centroids;                    #
#        Update centroids by computing the avg for each cluster         #
#        until either of the following satisfies:                       #
#           (1) all the centroids keep still.                           #
#           (2) iteration num satisfies.                                #
#                                                                       #
# Input file format (numerical):                                        #
#        feature1,  feature2,  ...,  label                              #
#        value11,   value21,   ...,                                     #
#        value12,   value22,   ...,                                     #
#        value13,   value23,   ...,                                     #
#          ...        ...      ...                                      #
#                                                                       #
# Input cmd args:                                                       #
#        --k=? --centroids='[(centroid1),(centroid2),...,(centroidk)]'  #
#                                                                       #
# Output format:                                                        #
#        Iteration 1:                                                   #
#           [centroid1, centroid2, ..., centroidk]                      #
#        Iteration 2:                                                   #
#           [centroid1, centroid2, ..., centroidk]                      #
#        Iteration 3:                                                   #
#           [centroid1, centroid2, ..., centroidk]                      #
#                   ...                                                 #
#        The last iteration would be the final centroids.               #
# ##################################################################### #

from mrjob.job import MRJob
import numpy
import sys

centroids = []

def get_centroid(job, runner):
    return centroids

def get_arg(args, target):
    value = 0
    for arg in args:
        if target in arg:
            value = int(arg.split('=')[1])
            break
    return value

class kmeans(MRJob):
    centroids = []
    def configure_options(self):
        super(kmeans, self).configure_options()
        self.add_passthrough_option('--k', dest='k', type='int', help='number of clusters')
        self.add_passthrough_option('--iteration', type='int', help='number of iterations')
        self.add_passthrough_option('--centroids', dest='centroids', default='[(25,1,226802,3,7,3,8,2,5,2,0,0,40,1),(38,1,89814,4,9,1,10,3,1,2,0,0,50,1)]')

    def assign_cluster(self, _, line):
        global centroids
        line = line.split(',')
        line = [int(i) for i in line]
        point = numpy.array(line)
        centroids = eval(str(self.options.centroids))
        k = int(self.options.k)
        distances = [numpy.linalg.norm(point - c) for c in centroids]
        min_dist = numpy.argmin(distances)
        yield centroids[min_dist], point.tolist()

    def sum_cluster(self, cluster, points):
        n = 1
        point_sum = numpy.array(points.next())
        for point in points:
            point_sum += point
            n += 1
        yield cluster, (point_sum.tolist(), n)

    def average_cluster(self, cluster, point_sum_n):
        point_sum, num = point_sum_n.next()
        point_sum = numpy.array(point_sum)
        for p_s,n in point_sum_n:
            point_sum += p_s
            num += n
        new_centroid = tuple(point_sum / float(num))
        centroids.append(new_centroid)


    def steps(self):
        return[
            self.mr(mapper=self.assign_cluster,
                    combiner=self.sum_cluster,
                    reducer=self.average_cluster)
        ]


if __name__ == '__main__':
    args = sys.argv[1:]
    args = list(args)
    k = get_arg(args, '--k')
    iterations = get_arg(args, '--iteration')
    kmeans.run()
    pre_centroids = []
    centroids = centroids[k*(-1):]

    count = 1
    while pre_centroids!=centroids and count<=iterations:
        print 'Iteration {0}'.format(count)
        pre_centroids = centroids
        get_centroid_job = kmeans(args=args+['--centroids='+str(centroids)])
        with get_centroid_job.make_runner() as get_centroid_runner:
            get_centroid_runner.run()
        centroids = get_centroid(get_centroid_job, get_centroid_runner)[k*(-1):]
        print centroids
        count += 1