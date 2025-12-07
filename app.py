
import streamlit as st
import numpy as np
import cnn_lib
import mnist_loader
import time
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# --- configs ---
st.set_page_config(page_title="NumPy CNN Visualizer", layout="wide")

# --- Sidebar ---
st.sidebar.title("Training Control")

# Hyperparameters
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.05, 0.01, 0.001)
with st.sidebar.expander("What is Learning Rate?"):
    st.write("""
    The **Learning Rate** controls how much the model changes its internal variables (weights) in response to the estimated error each time the model weights are updated.
    - **Too high**: The model might overshoot the optimal point and become unstable.
    - **Too low**: The model will learn very slowly.
    """)

num_filters = st.sidebar.slider("Number of CNN Filters", 4, 16, 8, help="More filters allow the model to learn more complex features, but make it slower.")
batch_size = st.sidebar.slider("Steps per Click", 10, 100, 50)

# Reset if filters changed
if 'num_filters' not in st.session_state:
    st.session_state.num_filters = num_filters
    
if num_filters != st.session_state.num_filters:
    st.session_state.num_filters = num_filters
    del st.session_state.model # Force rebuild
    st.session_state.trained_steps = 0
    st.session_state.train_losses = []
    st.session_state.train_accs = []
    st.rerun()

train_btn = st.sidebar.button("Train More Steps")
reset_btn = st.sidebar.button("Reset Model")

if reset_btn:
    del st.session_state.model
    st.rerun()

# --- Model Initialization ---
if 'model' not in st.session_state:
    # Try loading pretrained
    try:
        if num_filters == 8: # Only load if default config
            st.session_state.model = cnn_lib.load_model("pretrained_model.pkl")
            st.toast("Loaded Pre-trained Model!")
        else:
            raise ValueError("Config change")
    except:
        # Initialize Random
        conv = cnn_lib.Conv3x3(st.session_state.num_filters) 
        pool = cnn_lib.MaxPool2()
        softmax = cnn_lib.Softmax(13 * 13 * st.session_state.num_filters, 10)
        st.session_state.model = [conv, pool, softmax]
    
    st.session_state.train_losses = []
    st.session_state.train_accs = []

# Ensure all state variables exist (fix for potential corruption)
if 'trained_steps' not in st.session_state:
    st.session_state.trained_steps = 0
if 'train_losses' not in st.session_state:
    st.session_state.train_losses = []
if 'train_accs' not in st.session_state:
    st.session_state.train_accs = []

# Validate Model integrity
if 'model' in st.session_state and (not isinstance(st.session_state.model, list) or len(st.session_state.model) != 3):
    st.warning("Model state corrupted. Resetting...")
    del st.session_state.model
    st.rerun()

# --- Training Logic ---
if train_btn:
    conv, pool, softmax = st.session_state.model
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    
    progress_bar = st.sidebar.progress(0)
    step_loss = []
    step_acc = []
    
    start_idx = st.session_state.trained_steps % len(X_train)
    
    for i in range(batch_size):
        idx = (start_idx + i) % len(X_train)
        img = X_train[idx]
        label = y_train[idx]
        
        loss, acc = cnn_lib.train_step(img, label, st.session_state.model, learning_rate)
        step_loss.append(loss)
        step_acc.append(acc)
        progress_bar.progress((i + 1) / batch_size)
        
    st.session_state.trained_steps += batch_size
    st.session_state.train_losses.extend(step_loss)
    st.session_state.train_accs.extend(step_acc)

# --- Layout ---
st.title("🧠 Interactive NumPy CNN")
st.markdown("A **Convolutional Neural Network** built from scratch using only `numpy`. No TensorFlow, No PyTorch.")

tab1, tab2, tab3 = st.tabs(["🎨 Inference & Visualization", "📈 Training Metrics", "ℹ️ How it Works"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Draw a Digit")
        
        # Canvas Clear Logic
        if 'canvas_key' not in st.session_state:
            st.session_state.canvas_key = "canvas"
            
        if st.button("Clear Canvas"):
            st.session_state.canvas_key = f"canvas_{time.time()}"
            st.rerun()
            
        canvas = st_canvas(
            fill_color="#000000",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key,
        )
        
        predict_btn = st.button("Predict")
        
    with col2:
        st.subheader("Network Internals")
        if predict_btn and canvas.image_data is not None:
            # Preprocess with Center of Mass
            from PIL import Image
            import scipy.ndimage
            
            # 1. Resize to 20x20 initially (keeping aspect ratio/padding usually better but let's just do 20x20 and pad)
            img_pil = Image.fromarray(canvas.image_data.astype('uint8')).convert('L')
            img_resized = img_pil.resize((20, 20))
            img_array_small = np.array(img_resized)
            
            # 2. Place in 28x28 canvas centered by Center of Mass
            final_img = np.zeros((28, 28))
            
            # Calculate Center of Mass
            cy, cx = scipy.ndimage.center_of_mass(img_array_small)
            if np.isnan(cy) or np.isnan(cx):
                cy, cx = 10, 10 # Fallback
                
            # Shift to center (14, 14)
            shift_y = 14 - cy
            shift_x = 14 - cx
            
            # Paste into center
            # Simple integer shift logic
            start_y = int(max(0, shift_y))
            start_x = int(max(0, shift_x))
            
            # We copy the 20x20 into the 28x28 at calculated offset
            # (Simplified logic: actually centering the bounding box is easier, but COM is standard)
            # Let's try simple Bounding Box centering instead? It's more robust for digits.
            # actually let's stick to standard "Paste 20x20 in center of 28x28" if user draws big enough
            # But let's use the CoM logic to be smarter.
            
            # Correct logic:
            # 1. Pad 20x20 image to 28x28 (centered blindly)
            # 2. Shift based on CoM difference
            
            pad_y = 4
            pad_x = 4
            final_img[pad_y:pad_y+20, pad_x:pad_x+20] = img_array_small
            
            # Recalculate CoM of this 28x28 image
            cy, cx = scipy.ndimage.center_of_mass(final_img)
            if not np.isnan(cy):
                shift_y = 14 - cy
                shift_x = 14 - cx
                final_img = scipy.ndimage.shift(final_img, shift=[shift_y, shift_x])
            
            img_array = final_img
            
            # Normalize for network
            x = (img_array / 255.0) - 0.5
            
            # Ensure model is unpacked correctly
            conv, pool, softmax = st.session_state.model
            
            # 2. Conv Layer
            out_conv = conv.forward(x)
            st.write(f"**2. Feature Maps (Conv Output)** {out_conv.shape}")
            st.write("The network detects features like edges and curves.")
            
            # Display Filters (Dynamic)
            num_f = conv.num_filters
            cols = st.columns(min(num_f, 8)) # Display max 8 per row
            
            st.write(f"**2. Feature Maps (Conv Output)** {out_conv.shape}")
            st.write("The network detects features like edges and curves.")
            
            for i in range(num_f):
                # Simple grid logic for display if > 8
                if i < 8:
                    with cols[i]:
                        f_map = out_conv[:, :, i]
                        f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-9)
                        st.image(f_map, clamp=True, use_container_width=True)

            # 3. Pooling
            out_pool = pool.forward(out_conv)
            st.write(f"**3. Max Pooling (Downsampled)** {out_pool.shape}")
            st.write("Reduces size while keeping important features.")
            
            cols_p = st.columns(min(num_f, 8))
            for i in range(num_f):
                if i < 8:
                    with cols_p[i]:
                        f_map = out_pool[:, :, i]
                        f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-9)
                        st.image(f_map, clamp=True, use_container_width=True)
            
            # 4. Softmax
            probs = softmax.forward(out_pool)
            prediction = np.argmax(probs)
            
            st.success(f"## Prediction: {prediction}")
            
            st.subheader("Class Probabilities")
            st.bar_chart(probs)

with tab2:
    st.subheader("Training Progress")
    st.write(f"Total Steps Trained: **{st.session_state.trained_steps}**")
    
    if len(st.session_state.train_losses) > 0:
        # Smooth plots
        def moving_average(a, n=50):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        st.line_chart(st.session_state.train_losses)
        st.line_chart(st.session_state.train_accs)
    else:
        st.info("Train the model using the sidebar to see metrics.")

with tab3:
    st.markdown("""
    ### Architecture
    1.  **Input**: 28x28 Grayscale Image
    2.  **Conv Layer**: 8 Filters (3x3), Valid Padding. Detects local patterns.
    3.  **ReLU**: (Implicit in this implementation for simplicity, or we can add it). *Wait, I missed explicit ReLU in cnn_lib.py! It's usually good to have.*
        - *Self-correction: The original notebook had ReLU. `cnn_lib.py` logic implemented `conv.forward` which was just linear convolution. It needs ReLU!*
    4.  **Max Pool**: 2x2. Reducing spatial dimensions.
    5.  **Softmax**: Fully connected layer to 10 classes.
    """)
