
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
st.sidebar.caption("v1.1.0 (Mobile Fix)")

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

# --- Data Loading (Ensure it exists) ---
if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
    try:
        # Load a subset for responsiveness, or full set if possible
        # We'll load 1000 images for the web app to be fast
        X_all, y_all = mnist_loader.load_mnist(num_images=2000) 
        st.session_state.X_train = X_all
        st.session_state.y_train = y_all
        st.toast("MNIST Data Loaded!")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.session_state.X_train = []
        st.session_state.y_train = []

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
    
    if len(X_train) == 0:
        st.error("Training data not loaded. Cannot train.")
    else:
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
            
            # Capture last example for visualization (only on last step of batch)
            if i == batch_size - 1:
                st.session_state.last_train_img = img
                st.session_state.last_train_label = label
                # Quick forward pass to see what it *would* predict (or rather what it did)
                # Since train_step returns loss/acc but not pred, let's just cheat and infer
                # Actually acc is 1 if correct.
                # Let's do a quick forward for robust display
                c, p, s = st.session_state.model
                # Forward (re-run)
                out = c.forward(img)
                out = cnn_lib.relu(out)
                out = p.forward(out)
                probs = s.forward(out)
                st.session_state.last_train_pred = np.argmax(probs)
            
        st.session_state.trained_steps += batch_size
        # Safe update of state
        if 'train_losses' not in st.session_state:
            st.session_state.train_losses = []
        if 'train_accs' not in st.session_state:
            st.session_state.train_accs = []
            
        st.session_state.train_losses.extend(step_loss)
        st.session_state.train_accs.extend(step_acc)
        
        st.toast(f"Trained {batch_size} steps! Avg Loss: {np.mean(step_loss):.4f}")

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
        
        st.caption("Draw a digit above to see Real-time Prediction.")
        
    with col2:
        st.subheader("Network Internals")
        if canvas.image_data is not None:
            # Preprocess with Center of Mass
            from PIL import Image
            import scipy.ndimage
            
            # 1. Resize to 20x20
            img_pil = Image.fromarray(canvas.image_data.astype('uint8')).convert('L')
            img_resized = img_pil.resize((20, 20))
            img_array_small = np.array(img_resized)
            
            # 2. Place in 28x28 canvas centered by Center of Mass
            final_img = np.zeros((28, 28))
            cy, cx = scipy.ndimage.center_of_mass(img_array_small)
            if np.isnan(cy) or np.isnan(cx):
                cy, cx = 10, 10
            
            shift_y = 14 - cy
            shift_x = 14 - cx
            
            pad_y = 4
            pad_x = 4
            final_img[pad_y:pad_y+20, pad_x:pad_x+20] = img_array_small
            
            cy, cx = scipy.ndimage.center_of_mass(final_img)
            if not np.isnan(cy):
                shift_y = 14 - cy
                shift_x = 14 - cx
                final_img = scipy.ndimage.shift(final_img, shift=[shift_y, shift_x])
            
            img_array = final_img
            x = (img_array / 255.0) - 0.5
            
            # Ensure model is unpacked correctly
            conv, pool, softmax = st.session_state.model
            
            # --- VISUALIZATION START ---
            
            st.write("### Step 0: The Input & Filters")
            col_in, col_filt = st.columns([1, 2])
            with col_in:
                st.image(img_array, width=100, caption="Input (28x28)", clamp=True)
            with col_filt:
                st.write("**The Filters (Weights)**")
                # Normalize filters for display
                filters = conv.filters
                f_cols = st.columns(min(conv.num_filters, 8))
                for i in range(min(conv.num_filters, 8)):
                   with f_cols[i]:
                       f_img = filters[i]
                       f_img = (f_img - f_img.min()) / (f_img.max() - f_img.min() + 1e-9)
                       st.image(f_img, width=50, clamp=True)
                st.caption("These 3x3 grids scan the image for matching patterns.")
            
            # 1. Convolution
            out_conv = conv.forward(x)
            st.divider()
            st.write("### Step 1: Convolution (Feature Extraction)")
            st.write("The filters slide over the image. High values (bright pixels) mean a match found.")
            
            with st.expander("Show Math 🧮"):
                st.latex(r"Output_{i,j} = \sum_{m=0}^{2}\sum_{n=0}^{2} Input_{i+m, j+n} \times Filter_{m,n}")
            
            cols = st.columns(min(conv.num_filters, 8))
            for i in range(min(conv.num_filters, 8)):
                with cols[i]:
                    f_map = out_conv[:, :, i]
                    f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-9)
                    st.image(f_map, clamp=True, use_container_width=True, caption=f"Map {i}")

            # 2. ReLU
            st.divider()
            st.write("### Step 2: Activation (ReLU)")
            out_relu = cnn_lib.relu(out_conv)
            st.write("Negative values are removed (set to 0) to introduce non-linearity.")
            
            with st.expander("Show Math 🧮"):
                st.latex(r"ReLU(x) = \max(0, x)")

            cols_r = st.columns(min(conv.num_filters, 8))
            for i in range(min(conv.num_filters, 8)):
                with cols_r[i]:
                    f_map = out_relu[:, :, i]
                    f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-9)
                    st.image(f_map, clamp=True, use_container_width=True)

            # 3. Pooling
            st.divider()
            st.write("### Step 3: Max Pooling (Downsampling)")
            out_pool = pool.forward(out_relu)
            st.write(f"The image size is reduced from {out_relu.shape[0]}x{out_relu.shape[1]} to {out_pool.shape[0]}x{out_pool.shape[1]}.")
            
            with st.expander("Show Math 🧮"):
                st.write("Takes the **maximum** value in every 2x2 window.")
            
            cols_p = st.columns(min(conv.num_filters, 8))
            for i in range(min(conv.num_filters, 8)):
                with cols_p[i]:
                    f_map = out_pool[:, :, i]
                    f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-9)
                    st.image(f_map, clamp=True, use_container_width=True)
            
            # 4. Softmax
            st.divider()
            st.write("### Step 4: Classification (Softmax)")
            probs = softmax.forward(out_pool)
            prediction = np.argmax(probs)
            
            st.success(f"## Prediction: **{prediction}**")
            st.bar_chart(probs)

with tab2:
    st.subheader("Training Inspection")
    
    # Init vars for inspection
    if 'last_train_img' not in st.session_state:
        st.session_state.last_train_img = None
        st.session_state.last_train_label = None
        st.session_state.last_train_pred = None
    
    col_metrics, col_visual = st.columns([1, 1])
    
    with col_metrics:
        st.metric("Total Steps", st.session_state.get('trained_steps', 0))
        
        train_losses = st.session_state.get('train_losses', [])
        if len(train_losses) > 0:
            avg_loss = np.mean(train_losses[-50:]) if len(train_losses) > 50 else np.mean(train_losses)
            st.metric("Recent Loss", f"{avg_loss:.4f}")
            
            st.line_chart(train_losses[-100:], height=150)
            st.caption("Loss Trend (Lower is better)")
            
    with col_visual:
        st.markdown("##### Last Training Example")
        if st.session_state.last_train_img is not None:
            # Show Image
            img = st.session_state.last_train_img
            label = st.session_state.last_train_label
            pred = st.session_state.last_train_pred
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(img, width=100, caption=f"Truth: {label}", clamp=True)
            with c2:
                if pred == label:
                    st.success(f"AI Guessed: {pred} ✅")
                else:
                    st.error(f"AI Guessed: {pred} ❌")
            
            st.info(f"The model saw this **{label}**, guessed **{pred}**, and updated its weights to reduce the error.")
        else:
            st.info("Train at least one step to inspect the process!")

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
