# ##################################################################### #
# File Desc: Naive Bayes MapReduce                                      #
# Author: Yixin Chen, University of Iowa.                               #
# Created: 10/01/2014                                                   #
# Python Version: 2.7                                                   #
# Note: this script implements Naive Bayes Algorithm using MapReduce.   #
# Steps: compute the frequency for each Label_feature_value pair,       #
#        Label_feature, Label, store them as key and their frequency    #
#        as value into dictionary.                                      #
# Input file format:                                                    #
#        feature1,  feature2,  ...,  label                              #
#        value11,   value21,   ...,  label1                             #
#        value12,   value22,   ...,  label2                             #
#        value13,   value23,   ...,  label2                             #
#          ...        ...      ...    ...                               #
# Output dictionary format:                                             #
#       {'Label1_feature1_value11': freq1,                              #
#        'Label1_feature2_value21': freq2,                              #
#        'Label1_feature1': freq3,                                      #
#        'Label1': freq4, ...}                                          #
# ##################################################################### #

from mrjob.job import MRJob

dict = {}

class naiveBayes(MRJob):

    features = []
    i = True
    records = []

    # ################################################################# #
    # Function: mapper1                                                 #
    # Input: This function reads in training set (including feature     #
    #         set, labels and values) from the input file.              #
    # Output:(label_feature_value, 1)                                   #
    # ################################################################# #
    def mapper1(self, _, line):
        temp = line.split(',')
        if self.i==True:
            # read in feature set
            self.features = temp
            self.i = False
        else:
            self.records = temp
            last = len(self.features)-1
            label = self.features[last]
            label_value = self.records[last]
            for idx in range(0, len(self.features)-1):
                yield label_value+'_'+self.features[idx]+'_'+self.records[idx],1


    # ################################################################# #
    # Function: reducer1                                                #
    # Output: 1. sum the occurrence of label_feature_value              #
    #            e.g. ["no_humidity_high", 2]                           #
    #         2. sum the occurrence of label                            #
    #            e.g. ["label_no", 2]                                   #
    # Note: the occurrence of labels should be divided by the           #
    #       total number of labels                                      #
    # ################################################################# #
    def reducer1(self, label_feature_value, occurrence):
        num = sum(occurrence)
        yes_no = 'label_'+label_feature_value.split('_')[0]
        if yes_no in dict.keys():
            dict[yes_no] += num
        else:
            dict[yes_no] = num
        yield None,(label_feature_value, num)

    # ################################################################# #
    # Function: mapper2, reducer2                                       #
    # Output: mapper2 yield the occurrence of label_feature,            #
    #         reducer2 sum them up for each label_feature pair          #
    # ################################################################# #
    def mapper2(self, key, value):
        label_feature_value = value[0]
        occur = value[1]
        dict[label_feature_value] = occur
        # e.g. "no_humidity_high"	4
        # extract "no_humidity" from "no_humidity_high"
        # and yield "no_humidity", 4
        l_f = '_'.join(label_feature_value.split('_')[:-1])
        yield l_f, occur

    def reducer2(self, label_feature, occurrence):
        dict[label_feature] = sum(occurrence)

    def steps(self):
        return [
            self.mr(mapper=self.mapper1,
                    reducer=self.reducer1),
            self.mr(mapper=self.mapper2,
                    reducer=self.reducer2),
        ]

if __name__=='__main__':
    naiveBayes.run()
    print dict


# ############################################################################################# #
# Use Case:                                                                                     #
# Input file format:                                                                            #
#        feature1,  feature2,  ...,  label                                                      #
#        value11,   value21,   ...,                                                             #
#        value12,   value22,   ...,                                                             #
#        value13,   value23,   ...,                                                             #
#          ...        ...      ...,                                                             #
#                                                                                               #
# for each input line, calculate:                                                               #
#                                                                                               #
# P_label1 = P(label=label1 | (feature1=value11,feature2=value21,...))                          #
#   proportional to                                                                             #
#   P((feature1=value11)|label=label1)*P((feature2=value21)|label=label1)* ... *P(label=label1) #
#                                                                                               #
# P_label2 = P(label=label2 | (feature1=value11,feature2=value21,...))                          #
#   proportional to                                                                             #
#   P((feature1=value11)|label=label2)*P((feature2=value21)|label=label2)* ... *P(label=label2) #
#                                                                                               #
# Note: P((feature?=value??)|label=label?)=dict[label?_feature?_value??]/dict[label?_feature?]  #
#       P(label=label1)=dict[label1]/(sum of values of all labels in dict)                      #
#                                                                                               #
# P_label1 = P_label1/(P_label1+P_label2)                                                       #
# P_label2 = P_label2/(P_label1+P_label2)                                                       #
#                                                                                               #
# Output format:                                                                                #
#        value11,   value21,   ...,   label1  0.9879                                            #
#        value12,   value22,   ...,   label2  0.8321                                            #
#        value13,   value23,   ...,   label2  0.9904                                            #
# ############################################################################################# #

