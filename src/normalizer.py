import numpy

class Normalizer:
    def __init__(self,training_data=[],test_data=[]):
        self.original_training_data = training_data
        self.original_test_data = test_data
        # self.min = numpy.ndarray.min(training_data)
        combined_data = numpy.concatenate((training_data,test_data),axis=0)
        self.min = numpy.ndarray.min(combined_data)
        self.max = numpy.ndarray.max(combined_data)
        self.range = self.max - self.min
        self.normed_training_data = None
        self.normed_test_data = None

        # import pdb; pdb.set_trace()
        True

    def training_data(self):
        if self.normed_training_data is None:
            self.normed_training_data = self.normalize(self.original_training_data)

        return self.normed_training_data

    def test_data(self):
        if self.normed_test_data is None:
            self.normed_test_data = self.normalize(self.original_test_data)

        return self.normed_test_data

    def normalize(self,data):
        # return (data - self.min)/float(self.range)
        return(data)
