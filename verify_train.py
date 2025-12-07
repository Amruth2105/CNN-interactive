
import numpy as np
import cnn_lib
import mnist_loader

def verify_training():
    print("Loading MNIST Data (via Custom Loader)...")
    (train_images, train_labels), (test_images, test_labels) = mnist_loader.load_mnist()

    # Use a small subset for verification speed
    train_images = train_images[:1000]
    train_labels = train_labels[:1000]

    print("Initializing Model...")
    conv = cnn_lib.Conv3x3(8)
    pool = cnn_lib.MaxPool2()
    # 28x28 -> 26x26 -> 13x13 -> 13*13*8 = 1352
    softmax = cnn_lib.Softmax(13 * 13 * 8, 10) 
    layers = [conv, pool, softmax]

    print("Starting Training (First 100 images)...")
    
    loss_history = []
    
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i >= 100: break
        
        # Train Step
        loss, acc = cnn_lib.train_step(im, label, layers)
        loss_history.append(loss)
        
        if i % 10 == 0:
            avg_loss = np.mean(loss_history[-10:]) if len(loss_history) > 0 else loss
            print(f"Step {i}: Loss {loss:.4f} | Avg Loss (last 10): {avg_loss:.4f} | Acc: {acc}")

    print("Training verification complete.")

if __name__ == "__main__":
    verify_training()
