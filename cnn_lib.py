
import numpy as np

# --- Layers ---
class Conv3x3:
    """
    A simple 3x3 Convolution Layer using only NumPy.
    Assumes fixed kernel size of 3x3.
    """
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Filters: (num_filters, 3, 3) / Xavier Initialization
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        """
        Generates all possible 3x3 image regions using 'valid' padding.
        - image: 2D numpy array
        """
        h, w = image.shape
        # Valid padding means output dim will be reduced by 2
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        """
        Performs 3x3 Convolutions.
        - input: 2D numpy array (H, W)
        Returns: 3D numpy array (num_filters, H-2, W-2)
        """
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # Element-wise multiply region by filters and sum
            # axis=(1,2) ensures we sum over the 3x3 region for each filter
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backward(self, d_L_d_out, learn_rate):
        """
        Backpropagation for Conv3x3.
        - d_L_d_out: Gradient of Loss w.r.t Output of this layer.
                     Shape: (H-2, W-2, num_filters)
        """
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters
        
        # NOTE: We do NOT return gradients to iterate further back in this simple version
        # because this is the first layer. If we added more Conv layers, we would need
        # to calculate and return d_L_d_input.
        return None

class MaxPool2:
    """
    A Max Pooling Layer with a fixed 2x2 kernel size.
    """
    def iterate_regions(self, image):
        """
        Generates non-overlapping 2x2 image regions to pool over.
        - image: 3D numpy array (H, W, num_filters)
        """
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        """
        Performs Forward Pass of Max Pooling.
        - input: 3D numpy array (H, W, num_filters)
        """
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        """
        Performs Backward Pass of Max Pooling.
        - d_L_d_out: Gradient of loss w.r.t this layer's output.
        """
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max, pass the gradient
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

class Softmax:
    """
    A Standard Fully-Connected Layer with Softmax activation.
    input_len: Number of flattened input nodes.
    nodes: Number of output nodes (e.g., 10 for MNIST).
    """
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        """
        Performs forward pass with Softmax.
        - input: 3D numpy array (H, W, Filters) -> flattened.
        """
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        # Softmax
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backward(self, d_L_d_out, learn_rate):
        """
        Backpropagation for Softmax layer.
        - d_L_d_out: Gradient of Loss w.r.t Probabilities
        """
        # We assume d_L_d_out passed here is `d_L_d_t` (Gradient w.r.t totals)
        # which is (prediction - target) for Cross Entropy Loss.
        d_L_d_t = d_L_d_out 

        d_t_d_w = self.last_input
        d_t_d_b = 1
        d_t_d_inputs = self.weights

        # Gradients of Loss against Weights/Biases/Input
        d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
        d_L_d_b = d_L_d_t * d_t_d_b
        d_L_d_inputs = d_t_d_inputs @ d_L_d_t

        # Update weights / biases
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)

# --- Helpers ---

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def forward(image, label, layers):
    """
    Completes a forward pass of the CNN and calculates loss and accuracy.
    - layers: [conv, pool, softmax]
    """
    conv, pool, softmax = layers
    
    # Forward
    # 1. Conv
    out_conv = conv.forward((image / 255) - 0.5)
    # 2. ReLU
    out_relu = relu(out_conv)
    # 3. MaxPool
    out_pool = pool.forward(out_relu)
    # 4. Softmax
    probs = softmax.forward(out_pool)

    # Loss: Cross-Entropy Loss = -ln(probability of correct class)
    loss = -np.log(probs[label])
    
    # Accuracy: 1 if correct, 0 if not
    acc = 1 if np.argmax(probs) == label else 0

    return out_conv, out_relu, out_pool, probs, loss, acc

def train_step(image, label, layers, alpha=0.005):
    """
    Completes a full training step (forward + backward) for a single image.
    """
    conv, pool, softmax = layers
    
    # Forward
    out_conv, out_relu, out_pool, probs, loss, acc = forward(image, label, layers)

    # Calculate Gradient for Softmax
    # dL/dt = p - OneHot(y)
    gradient = np.zeros(10)
    gradient[label] = -1
    d_L_d_t = probs + gradient 

    # Backward
    grad_softmax = softmax.backward(d_L_d_t, alpha)
    grad_pool = pool.backward(grad_softmax)
    
    # ReLU derivative
    grad_relu = grad_pool * relu_deriv(out_conv)
    
    grad_conv = conv.backward(grad_relu, alpha)

    return loss, acc

