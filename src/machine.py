class Machine:
    """
    Represents an invididual machine in the network. We will instantiate it with some data,
    ask it to perform some gradient updates and return the results, then forget about this
    machine and the data that was given to it.
    """
    def __init__(self, source, data_count):
        X, Y = source.generate_observations(data_count)
        self._X = X # Locally stored predictor data
        self._Y = Y # Locally stored responder data

    def get_stochastic_gradient(self, sample):
        pass # TODO: return the stochastic gradient estimate evaluated on sample (list of indices)

    def execute_local_sgd(self, k, eta_local, w0):
        pass # TODO: perform sequential gradient steps and return relevant data

    def execute_modified_DSVRG(self, w_cur, w_prev, g_prev):
        pass # TODO: perform gradient steps and return relevant data