import numpy as np

class Bernoulli:
  """
  Bernoulli discrete random variable

  Controlled by theta in [0,1]
  P(X = 0) = 1 - theta
  P(X = 1) = theta
  where X is a random variable.
  """
  def __init__(self):
    self.sample_ = None
    self.pmf_ = None
    self.expectation_ = None
    self.var_ = None

  #Used to genrate a bernoulli random variable
  def generate(self, theta, size):
      """
      Generates a bernoulli random variable

      Arguments:
      theta - shape parameter
      size - size of the array to be returned

      """
    #S1. Generate an array guided by theta
    ones = np.ones(int(size*theta))
    #S2. Generate an array guided by 1-theta
    zeros = np.zeros(int(size*(1-theta)))
    #S3. Merge the two arrays
    sample = np.concatenate((ones,zeros), axis = None) #axis = None helps flatten before joining
    np.random.shuffle(sample)
    self.sample_ = sample

  def expectation(self, input):
      """
      Computes the expectation of any bernoulli random variable

      Arguments:
      input - The input array for which expectation is required to be computed
      """
    self.expectation_ = list(input).count(1)/input.shape[0] #see the class docstring for information on why 1 is used here

  def pmf(self, input):
      """
      Computes the PMF of any bernoulli random variable

      Arguments:
      input - Input numpy array for which PMF is required
      """
    self.expectation(input)
    #Write function to compute PMF
    def pmf_estimation(input_element):
      return self.expectation_**input_element * (1-self.expectation_)**(1 - input_element)
      #Apply pmf function to each element of the array 
    self.pmf_ = np.apply_along_axis(pmf_estimation, 0, input)

  def var(self, input):
      """
      Computes the variance of any bernoulli random variable

      Arguments:
      input - The input array for which variance is required
      """
    self.expectation(input)
    self.var_ = self.expectation_ * (1 - self.expectation_)
