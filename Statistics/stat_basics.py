

class VisualFunctions:

  def __init__(self):
    self.x_axis_ = None
    self.y_axis_ = None

  def plot_histogram(self, input_data, bin_size = 5):
    """
    Plot's a histogram for given data.

    Arguments:
    input_data: Real numbers
    """
    #Sort input data
    sorted_input = sorted(input_data) ###Write own sortin routine
    #calculate min/max of sorted_input
    min_sorted_input = int(sorted_input[0])
    max_sorted_input = int(sorted_input[-1])
    #break input data into buckets and use as x-axis
    self.x_axis_ = [i for i in range(min_sorted_input, max_sorted_input+1, bin_size)]
    #calculate no. of values in each bin/cardinality of each bin to use as y axis
    bin_iter = 0
    cardinality_bin = [[] for i in self.x_axis_]
    sorted_input_copy = sorted_input.copy()
    for i in self.x_axis_:
      cardinality_bin[bin_iter] = len([j for j in sorted_input_copy if j<i+bin_size])
      sorted_input_copy = [k for k in sorted_input_copy if k >= i+bin_size]
      bin_iter +=1
    self.y_axis_ = cardinality_bin

#Example use case

import numpy as np
import matplotlib.pyplot as plt


vf1 = VisualFunctions()
vf2 = VisualFunctions()

sample1 = [i for i in range(100)] #uniform distribution
vf1.plot_histogram(sample1)
plt.plot(vf1.x_axis_, vf1.y_axis_)

sample2 = np.random.normal(200, 5, 10000).tolist() #gaussian
vf2.plot_histogram(sample2, bin_size=2)
plt.plot(vf2.x_axis_, vf2.y_axis_)
