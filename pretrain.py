
import numpy as np
import cnn_lib
import mnist_loader
import time

def pretrain():
    print("Loading Full MNIST Data...")
    (X_train, y_train), (X_test, y_test) = mnist_loader.load_mnist()
    
    # Normalize
    X_train = X_train[:5000] # Train on 5000 examples (full 60k is too slow for pure numpy in this demo context)
    y_train = y_train[:5000]
    
    print(f"Training on {len(X_train)} images...")
    
    # Initialize Model
    conv = cnn_lib.Conv3x3(8)
    pool = cnn_lib.MaxPool2()
    softmax = cnn_lib.Softmax(13 * 13 * 8, 10)
    layers = [conv, pool, softmax]
    
    start_time = time.time()
    loss_history = []
    
    for i, (im, label) in enumerate(zip(X_train, y_train)):
        loss, acc = cnn_lib.train_step(im, label, layers, alpha=0.01)
        loss_history.append(loss)
        
        if i % 100 == 0:
            avg = np.mean(loss_history[-100:]) if len(loss_history) > 0 else loss
            print(f"Step {i}/{len(X_train)} | Loss: {avg:.4f} | Time: {time.time() - start_time:.1f}s")
            
    print("Training Complete.")
    cnn_lib.save_model(layers, "pretrained_model.pkl")

if __name__ == "__main__":
    pretrain()
