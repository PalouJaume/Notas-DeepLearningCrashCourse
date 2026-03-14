from numpy.random import default_rng
from loader import load_data_1d


def neuron_clas_1d(w0, x):
    """Artificial neuron for 1D classification."""
    return (w0 * x > 0).astype(int)

def initialize_neuron():
    """Initialize Artificial neuron for 1D classification"""
    rng = default_rng()
    w0 = rng.standard_normal()
    print("[INFO] Initial guess :", w0)
    return w0

def train_neuron(it, lr, filename):
    """Train Artificial neuron for 1D classification"""
    print(f"[INFO] numero de iteraciones={it} learning rate={lr}")
    (x, y_gt) = load_data_1d(filename)
    
    num_samples = len(x)
    num_train_iterations = it
    eta = lr  # Learning rate
    rng = default_rng()

    print(f"[INFO] numero de iteraciones={it} learning rate={lr}")

    w0 = initialize_neuron()

    for i in range(num_train_iterations):
        selected = rng.integers(0, num_samples)  # Select random sample
        x0_selected = x[selected]
        y_gt_selected = y_gt[selected]
        y_p_selected = neuron_clas_1d(w0, x0_selected)  # Neuron prediction

        error = y_p_selected - y_gt_selected  # Calculate error

        w0 = w0 - eta * error * x0_selected  # Update neuron weight

        print(f"[INFO] i={i} w0={w0:.2f} error={error:.2f}")

    return w0