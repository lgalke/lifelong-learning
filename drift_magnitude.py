""" Functions to compute concept drift magnitude 
- lga"""
import numpy as np
# from scipy.special import rel_entr
from scipy.stats import entropy

def total_variation(pk, qk, axis=0):
    """ Computes total variation distance (Levin 2008) as measure for concept drift (Webb 2018) """
    # Cast to numpy ndarray
    pk = np.asarray(pk)
    qk = np.asarray(qk)

    if pk.shape != qk.shape:
        raise ValueError("pk and qk be must have same shapes")

    # Compute total variation distance
    tv = 0.5 * np.sum(np.abs(pk - qk), axis=axis)

    return tv


def kl_divergence(pk, qk, axis=0):
    """ Computes total variation distance (Levin 2008) as measure for concept drift (Webb 2018) """
    # Cast to numpy ndarray
    pk = np.asarray(pk)
    qk = np.asarray(qk)

    if pk.shape != qk.shape:
        raise ValueError("pk and qk be must have same shapes")

    # Compute total variation distance
    return entropy(pk, qk, axis=axis)



VALID_METRICS = {'total_variation': total_variation,
                 'kl_divergence': kl_divergence}

def drift_magnitude(pk, qk, metric='total_variation', axis=0, **kwargs):
    """
    Computes the drift magnitude between pk and qk according to `metric`.
    Keyword arguments are passed to the metric function
    """
    # Cast to numpy ndarray
    pk = np.asarray(pk)
    qk = np.asarray(qk)

    # Normalize
    pk = 1.0 * pk / np.sum(pk, axis=axis, keepdims=True)
    qk = 1.0 * qk / np.sum(qk, axis=axis, keepdims=True)

    if callable(metric):
        return metric(pk, qk, **kwargs)

    return VALID_METRICS[metric](pk, qk, **kwargs)


def drift_magnitude_per_time(time, labels, t_start=None, history=None, cumulative=False, verbose=False,
        metric='total_variation'):
    time = np.asarray(time)
    labels = np.asarray(labels)
    assert time.shape == labels.shape
    assert history is None or history > 0, "History for previous data is exclusive, min 1"

    num_labels = len(np.unique(labels))
    if verbose:
        print("Found", num_labels, "globally unique label")

    steps = np.unique(time)
    if verbose:
        print("Found time steps between", steps[0], "and", steps[-1])

    if t_start is None:
        # If not given, start with step 1 (rather than 0)
        t_start = steps[1]

    eval_steps = steps[steps >= t_start]
    if verbose:
        print("Using time steps between", eval_steps[0], "and", eval_steps[-1])


    drift = []
    for eval_step in eval_steps:
        if history is None:
            # Assume infinite history
            previous_labels = labels[time < eval_step]
        else:
            # Use given history to select prev labels
            previous_labels = labels[(time < eval_step) & (time >= (eval_step - history))]

        # New data
        if cumulative:
            # Add history window also to current labels
            current_labels = labels[(time <= eval_step) & (time >= (eval_step - history))]
        else:
            current_labels = labels[time == eval_step]

        # Count labels
        previous_unique, previous_counts = np.unique(previous_labels, return_counts=True)
        current_unique, current_counts = np.unique(current_labels, return_counts=True)

        # Put counts into blank labels array (shapes must match!)
        pk = np.zeros(num_labels)
        pk[previous_unique] = previous_counts
        qk = np.zeros(num_labels)
        qk[current_unique] = current_counts

        # Compute drift magnitude
        dm = drift_magnitude(pk, qk, metric=metric)
        print(eval_step, ',', dm, sep='')
        drift.append(dm)

    return eval_steps, drift

def main():
    import argparse
    import seaborn
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('time', help="Path to numpy file with times")
    parser.add_argument('labels', help="Path to numpy file with labels")
    parser.add_argument('--t_start', default=None, type=int, help="Start time")
    parser.add_argument('--history', default=None, type=int, help="Sliding window size for previous labels")
    parser.add_argument('--cumulative', default=False, action='store_true', help="Apply history also to 'right-side'")
    parser.add_argument('--save_plot', default=None, type=str, help="Path to save plot")
    parser.add_argument('--info', default=False, action='store_true', help="Print some more info")
    args = parser.parse_args()
    verbose = args.info

    time = np.load(args.time)
    labels = np.load(args.labels)
    if verbose:
        print("Time shape", time.shape, "dtype", time.dtype)
        print("Labels shape", labels.shape, "dtype", labels.dtype)

    x, y = drift_magnitude_per_time(time, labels, t_start=args.t_start, verbose=verbose,
            history=args.history, cumulative=args.cumulative)

    seaborn.lineplot(x=x,y=y)
    if args.save_plot:
        plt.savefig(args.save_plot)
    else:
        plt.show()


if __name__ == "__main__":
    main()
