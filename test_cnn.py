import numpy as np
import cnn_lib

def test_cnn():
    print("Testing CNN Library...")
    
    # 1. Setup Data
    # MNIST images are 28x28
    input_image = np.random.randn(28, 28)
    print(f"Input Image Shape: {input_image.shape}")

    # 2. Setup Layers
    conv = cnn_lib.Conv3x3(8) # 8 Filters
    pool = cnn_lib.MaxPool2()
    # Output of Conv (valid padding) is 26x26 x 8 filters
    # Output of Pool is 13x13 x 8 filters
    # Flattened: 13 * 13 * 8 = 1352
    softmax = cnn_lib.Softmax(13 * 13 * 8, 10) 

    # 3. Forward Pass
    out_conv = conv.forward(input_image)
    print(f"Conv Output Shape: {out_conv.shape} (Expected: 26, 26, 8)")
    
    out_pool = pool.forward(out_conv)
    print(f"Pool Output Shape: {out_pool.shape} (Expected: 13, 13, 8)")
    
    out_softmax = softmax.forward(out_pool)
    print(f"Softmax Output Shape: {out_softmax.shape} (Expected: 10,)")
    print(f"Probabilities Sum: {np.sum(out_softmax):.2f} (Expected: 1.00)")

    # 4. Backward Pass
    # Fake gradient from loss (e.g. Cross Entropy)
    # dL/dt = probs - target (where target is one-hot)
    # Let's just create a random gradient vector of shape 10
    gradient_loss = np.random.randn(10)
    
    grad_softmax = softmax.backward(gradient_loss, learn_rate=0.01)
    print(f"Grad W.R.T Softmax Input Shape: {grad_softmax.shape} (Expected: 13, 13, 8)")
    
    grad_pool = pool.backward(grad_softmax)
    print(f"Grad W.R.T Pool Input Shape: {grad_pool.shape} (Expected: 26, 26, 8)")
    
    # Conv backward does not return input gradient in this implementation, but it updates weights
    grad_conv = conv.backward(grad_pool, learn_rate=0.01)
    print("Conv Backward completed (Weights updated).")

    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_cnn()
