import streamlit as st
import torch
import requests
import matplotlib.pyplot as plt
import time

from PIL import Image
from io import BytesIO
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Siglip2Processor,
    Siglip2Model,
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="siglip2-so400m-16-naflex Attention Map Visualizer")

# --- Constants ---
DEFAULT_CKPT = "google/siglip2-so400m-patch16-naflex"
DEFAULT_IMAGE_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg" # Bear
DEFAULT_PATCH_COUNTS = "256, 1024, 2304, 3136"
MODEL_LAYER_COUNT = 27 # For siglip2-so400m-patch16-naflex

# --- Caching Model/Processor ---
@st.cache_resource(show_spinner="Loading Siglip2 Model and Processor...")
def load_model_processor(ckpt_name):
    """Loads the model and processor, caches them."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_processor = AutoImageProcessor.from_pretrained(ckpt_name)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_name)
        processor = Siglip2Processor(image_processor=image_processor, tokenizer=tokenizer)
        model = Siglip2Model.from_pretrained(ckpt_name).to(device).eval()
        st.session_state.model_loaded = True
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model/processor '{ckpt_name}': {e}")
        st.session_state.model_loaded = False
        return None, None, None

# --- Helper Function for Processing Single Layer ---
def process_specific_attention_layer(raw_attn_tensor, h_patch, w_patch):
    """Averages heads and reshapes a single layer's raw attention tensor."""
    if raw_attn_tensor is None or h_patch <= 0 or w_patch <= 0:
        return None
    actual_patches = h_patch * w_patch
    # Process directly on CPU after detaching
    attn_weights = raw_attn_tensor.cpu().detach().mean(dim=1).squeeze(0)
    if actual_patches == 0 or attn_weights.shape[0] < actual_patches or attn_weights.shape[1] < actual_patches:
        return None
    if h_patch * w_patch != actual_patches:
         print(f"Warning: Mismatch h_patch*w_patch ({h_patch*w_patch}) vs actual_patches ({actual_patches})")
         return None
    avg_attn_received = attn_weights[:actual_patches, :actual_patches].mean(dim=0)
    attention_grid = avg_attn_received.numpy().reshape(h_patch, w_patch)
    return attention_grid

# --- Initialize Session State ---
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# --- Streamlit UI ---
st.title("siglip2-so400m-16-naflex Attention Map Visualizer")
st.markdown("Select an image, resolutions, and multiple layers, then click Generate. Computes attention on demand.")

# --- Load Model ---
model, processor, device = load_model_processor(DEFAULT_CKPT)

# --- Input Configuration ---
st.sidebar.header("Configuration")
input_method = st.sidebar.radio("Select Image Input Method", ('URL', 'Upload'), key='input_method')

image_url = None
uploaded_file = None

if input_method == 'URL':
    image_url = st.sidebar.text_input("Image URL", DEFAULT_IMAGE_URL)
else:
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

target_patches_str = st.sidebar.text_input(
    "Target Max Patch Counts (comma-separated)", DEFAULT_PATCH_COUNTS
)

# Layer Multi-Select
available_layers = list(range(MODEL_LAYER_COUNT))
default_layers = [5,6,7,8,9] # best what I observed
# Ensure defaults are valid
default_layers = [l for l in default_layers if l in available_layers]

layers_to_visualize = st.sidebar.multiselect(
    "Select Layers to Visualize",
    options=available_layers,
    default=default_layers,
    key="layer_multiselect",
    disabled=not st.session_state.get('model_loaded', False)
)


# --- Load and Display Input Image ---
image = None
load_error = False
if input_method == 'URL' and image_url:
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"URL Error: {e}")
        load_error = True
    except Exception as e:
        st.sidebar.error(f"Image Load Error: {e}")
        load_error = True
elif input_method == 'Upload' and uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.sidebar.error(f"File Load Error: {e}")
        load_error = True

# Display the loaded image
input_image_container = st.sidebar.empty()
if image:
    input_image_container.subheader("Input Image")
    input_image_container.image(image, caption="Loaded Image", use_container_width=True)
elif not load_error:
     input_image_container.info("Provide an image via URL or Upload.")

# --- Parse Other Inputs ---
valid_config = False
target_patch_counts_list = []
if st.session_state.model_loaded and not load_error and image and layers_to_visualize:
    try:
        target_patch_counts_list = sorted([
            int(p.strip()) for p in target_patches_str.split(",") if p.strip()
        ])
        if target_patch_counts_list:
            valid_config = True
        else:
            st.sidebar.warning("Enter target patch counts.")
    except ValueError:
        st.sidebar.error("Invalid patch counts. Use comma-separated integers.")
    if not layers_to_visualize: # Check if layers were selected
         st.sidebar.warning("Select at least one layer to visualize.")
         valid_config = False

# --- Generate Button and Visualization Logic ---
if st.sidebar.button("✨ Generate Visualizations", disabled=not valid_config):
    processing_log = [] # Log for this run
    attention_maps_data = {} # {resolution: {layer_index: map}}
    start_run_time = time.time()

    # --- Main Computation Loop ---
    with st.spinner(f"Generating attention for {len(layers_to_visualize)} layers across {len(target_patch_counts_list)} resolutions..."):
        for target_patches in target_patch_counts_list:
            log_entry = f"\nProcessing for target_patches = {target_patches}..."
            processing_log.append(log_entry)
            print(log_entry)
            attention_maps_data[target_patches] = {} # Init dict for this resolution

            try:
                # Preprocess for the current resolution
                batch = processor(
                    text="an image", images=image,
                    padding="max_length", truncation=True, max_length=64,
                    images_kwargs={"max_num_patches": target_patches},
                    return_tensors="pt",
                ).to(device)

                h_patch, w_patch = batch["spatial_shapes"][0].tolist()
                log_entry = f"  Grid: {h_patch}x{w_patch}, Padded: {batch['pixel_values'].shape[1]}"
                processing_log.append(log_entry)
                print(log_entry)

                # Run forward pass ONCE for this resolution
                with torch.no_grad():
                    vision_outputs = model.vision_model(
                        pixel_values=batch["pixel_values"],
                        attention_mask=batch["pixel_attention_mask"],
                        spatial_shapes=batch["spatial_shapes"],
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                # Extract and process ONLY the selected layers for THIS resolution
                attentions_all_layers = vision_outputs.attentions # Tuple of tensors (on device)
                log_entry = f"  Got {len(attentions_all_layers)} attention tensors for this resolution."
                processing_log.append(log_entry)
                print(log_entry)


                for layer_idx in layers_to_visualize:
                    if layer_idx < len(attentions_all_layers):
                        raw_attn_tensor = attentions_all_layers[layer_idx] # Get the raw tensor
                        # Process it (includes moving to CPU)
                        attention_grid = process_specific_attention_layer(raw_attn_tensor, h_patch, w_patch)
                        if attention_grid is not None:
                            attention_maps_data[target_patches][layer_idx] = attention_grid
                            # Log success minimally inside the inner loop
                            # print(f"    Processed L{layer_idx}")
                        else:
                            log_entry = f"  --> Failed processing L{layer_idx} for T:{target_patches}"
                            processing_log.append(log_entry)
                            print(log_entry)
                    else:
                        log_entry = f"  --> Layer index {layer_idx} out of bounds."
                        processing_log.append(log_entry)
                        print(log_entry)

                # Explicitly delete the large tensor tuple to potentially help GC, though Python might do it anyway
                del vision_outputs
                del attentions_all_layers
                if device == 'cuda':
                    torch.cuda.empty_cache() # Try to free GPU memory

            except Exception as e:
                error_msg = f"Error during processing for target_patches={target_patches}: {e}"
                processing_log.append(error_msg)
                print(error_msg)
                st.error(error_msg) # Show error in UI


    end_run_time = time.time()
    st.sidebar.info(f"Visualization run took {end_run_time - start_run_time:.2f} seconds.")

    # --- Plotting ---
    st.subheader("Attention Map Visualization")
    if attention_maps_data:
        num_resolutions = len(target_patch_counts_list)
        num_layers_viz = len(layers_to_visualize)

        # Adjust figure size dynamically
        fig_width = max(8, num_resolutions * 3.5)
        fig_height = max(4, num_layers_viz * 3) # Height depends on number of layers

        fig, axes = plt.subplots(
            num_layers_viz,
            num_resolutions,
            figsize=(fig_width, fig_height),
            squeeze=False # Always return 2D array for axes
        )

        plot_successful = False
        sorted_layers = sorted(layers_to_visualize) # Plot in layer order

        for i, layer_idx in enumerate(sorted_layers):
            for j, target_patches in enumerate(target_patch_counts_list):
                ax = axes[i, j]
                # Retrieve the potentially processed map
                attn_map = attention_maps_data.get(target_patches, {}).get(layer_idx)

                if attn_map is not None:
                    im = ax.imshow(attn_map, cmap="viridis")
                    ax.set_title(f"Layer {layer_idx}, Target: {target_patches}")
                    plot_successful = True
                else:
                    ax.set_title(f"L {layer_idx}, T: {target_patches}\n(Not Processed)")
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

                ax.axis("off")

        if plot_successful:
            plt.tight_layout()
            st.pyplot(fig)
            st.success("Visualization complete!")
        else:
            st.warning("Could not generate plots. Check logs.")

    else:
        st.warning("No attention maps were generated. Check logs.")

    # Display Log
    st.subheader("Processing Log")
    st.text("\n".join(processing_log))


elif not valid_config:
    st.info("👈 Configure valid inputs (Image, Patch Counts, Layers) in the sidebar and click 'Generate Visualization'.")