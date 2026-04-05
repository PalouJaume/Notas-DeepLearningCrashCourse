def neuron_clas_2d(w, x):
    """Artificial neuron for multidimensional classification."""
    return (x @ w > 0).astype(int)

def train_neuron_clas_2d(x, y_gt, rng, w, verbose):
    num_samples = len(x)
    num_train_iterations = 100
    eta = .1  # Learning rate

    for i in range(num_train_iterations):
        selected = rng.integers(0, num_samples)  # Select random sample
        x_selected = x[selected]
        y_gt_selected = y_gt[selected]

        y_p_selected = neuron_clas_2d(w, x_selected)  # Neuron prediction

        error = y_p_selected - y_gt_selected  # Calculate error

        w = w - eta * error * x_selected  # Update neuron weights

        if verbose:
            print(f"[INFO] i={i} w0={w[0]:.2f} w1={w[1]:.2f} error={error[0]:.2f}")

    return w