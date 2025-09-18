import math


def compute_alpha(rate, cutoff):
    """Calculate smoothing factor alpha."""
    tau = 1 / (2 * math.pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)


class LowPassFilter:
    """Simple low-pass filter using exponential smoothing."""
    def __init__(self):
        self.prev_value = None

    def apply(self, value, alpha):
        if self.prev_value is None:
            self.prev_value = value
            return value
        filtered = alpha * value + (1 - alpha) * self.prev_value
        self.prev_value = filtered
        return filtered


class OneEuroFilter:
    """One Euro Filter for smoothing signals with adaptive cutoff."""
    def __init__(self, freq=15, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.signal_filter = LowPassFilter()
        self.derivative_filter = LowPassFilter()

        self.prev_input = None

    def apply(self, value):
        # Compute derivative
        if self.prev_input is None:
            derivative = 0.0
        else:
            derivative = (value - self.prev_input) * self.freq

        self.prev_input = value

        # Smooth derivative
        alpha_d = compute_alpha(self.freq, self.d_cutoff)
        smoothed_derivative = self.derivative_filter.apply(derivative, alpha_d)

        # Compute dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)

        # Smooth signal
        alpha = compute_alpha(self.freq, cutoff)
        return self.signal_filter.apply(value, alpha)


if __name__ == '__main__':
    euro_filter = OneEuroFilter(freq=15, beta=0.1)
    for i in range(10):
        noisy_input = i + (-1)**(i % 2)  # add alternating noise
        smoothed_output = euro_filter.apply(noisy_input)
        print(f"Filtered: {smoothed_output:.4f}, Raw: {noisy_input}")
