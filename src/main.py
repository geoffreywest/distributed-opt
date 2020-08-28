###
### Platform for experimenting with distributed model fitting.
###

"""
Program goals
1. Experiment with inner/outer step size method
2. Experiment with modified DSVRG as outlined in Google Doc

Implementation approach
- Import dataset for logistic regression fitting
- Establish some baselines:
    - Generic SGD
    - Generic minibatch SGD
- Test 1. and 2.
"""

import numpy as np
import machine
import data_source

def run_method_1():
    # TODO
    pass

def run_method_2():
    # Replicates the method outlined in the Google Doc
    source = data_source.DataSource()
    T = 100
    K = 10

    zero_vec = np.zeros(10)
    ws = [zero_vec, zero_vec]
    gs = [zero_vec]
    for t in range(T):
        w_cur = ws[-1]
        w_prev = ws[-2]
        g_prev = gs[-1]

        # Generate K random machines to perform the modified DSVRG updates
        inner_results = [
            machine.Machine(source, data_count=100).execute_modified_DSVRG(w_cur, w_prev, g_prev)
        for _ in range(K)]
        w_next = np.mean([w for (w, _) in inner_results])
        g_cur = np.mean([g for (_, g) in inner_results])

        ws.append(w_next)
        gs.append(g_cur)
    pass

def main():
    run_method_1()
    run_method_2()
    return

if __name__ == '__main__':
    main()
