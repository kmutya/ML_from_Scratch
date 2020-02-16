import numpy as np

class Bernoulli:
  """
  Bernoulli discrete random variable

  Controlled by shape parameter theta in [0,1]
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



  #Generate a Discrete Uniform random variable
  def generate(self, k, size):
    """
    Generate a unifrom discrete random variable

    Arguments:
    k - the no. of states the random variable can take
    size - size of the array to be returned
    """
    states = np.arange(0,k)
    sample = np.empty((k,int(size/k))) #should have state equivalent rows and repetation equivalent columns
    for i in states:
      sample[i] = np.full((int(size/k)), i)
      i += 1
    sample = sample.flatten()
    self.sample_ = sample.astype(int)
    np.random.shuffle(self.sample_)

  def expectation(self, input):
    """
    Computes the expectation of any uniform random variable.
    More here: https://proofwiki.org/wiki/Expectation_of_Discrete_Uniform_Distribution

    Arguments:
    input - The int input array for which expectation is required to be computed
    """

    #Compute no. of unique states of input
    self.count_states_ = np.bincount(input).shape[0]
    self.expectation_ = (self.count_states_ + 1)/2

  def var(self, input):
    """
    Computes the variance of any uniform random variable.
    More here: https://proofwiki.org/wiki/Variance_of_Discrete_Uniform_Distribution

    Arguments:
    input - The int input array for which variance is required to be computed
    """

    self.expectation(input)
    self.var_ = (self.count_states_**2 - 1) / 12

  def pmf(self, input):
     """
     Computes the PMF of any discrete uniform random variable

     Arguments:
     input - Input numpy array for which PMF is required
     """
    self.expectation(input)
    self.pmf_ = np.full((input.shape[0]),1/self.count_states_)
