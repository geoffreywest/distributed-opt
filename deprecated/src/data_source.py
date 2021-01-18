class DataSource:
    """
    Acts like a "button" for random sample data. Reads in the entire sample dataset and stores it locally.
    Then returns data randomly to each machine to prevent re-use of the same data.
    """
    def __init__(self):
        # TODO: Read in data set
        self._X = None
        self._Y = None
        self._num_used = 0
        # TODO: Randomly order the data

    def generate_observations(self, count):
        X = self._X[self._num_used:self._num_used + count]
        Y = self._Y[self._num_used:self._num_used + count]
        self._num_used += count
        return (X, Y)