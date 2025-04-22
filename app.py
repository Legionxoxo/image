# app.py - Streamlit Image Colorization App
import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from urllib.request import urlretrieve
import requests
import io

st.set_page_config(
    page_title="Image Colorization App",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("📸 Grayscale Image Colorization")
st.markdown("""
This app uses deep learning to colorize black and white photos. Upload your image and watch it transform!
""")

# Function to download models
@st.cache_resource
def download_models():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with st.spinner("Downloading model files (first run only)..."):
        # Download the model files if they don't exist
        prototxt = 'models/colorization_deploy_v2.prototxt'
        caffemodel = 'models/colorization_release_v2.caffemodel'
        pts_npy = 'models/pts_in_hull.npy'
        
        # Updated URLs and download method for more reliability
        prototxt_url = 'https://github.com/richzhang/colorization/raw/caffe/colorization/models/colorization_deploy_v2.prototxt'
        caffemodel_url = 'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1'
        pts_npy_url = 'https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy'
        
        # Download prototxt file
        if not os.path.exists(prototxt):
            try:
                response = requests.get(prototxt_url)
                response.raise_for_status()
                with open(prototxt, 'wb') as f:
                    f.write(response.content)
                st.success(f"Downloaded {prototxt}")
            except Exception as e:
                st.error(f"Error downloading prototxt file: {e}")
                # Fallback to local creation if download fails
                create_prototxt_file(prototxt)
        
        # Download caffemodel file
        if not os.path.exists(caffemodel):
            try:
                response = requests.get(caffemodel_url)
                response.raise_for_status()
                with open(caffemodel, 'wb') as f:
                    f.write(response.content)
                st.success(f"Downloaded {caffemodel}")
            except Exception as e:
                st.error(f"Error downloading caffemodel: {e}")
                st.error("The model file is required and cannot be created locally.")
                st.info("You can manually download it from: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1")
                return None, None, None
        
        # Download pts_in_hull.npy file
        if not os.path.exists(pts_npy):
            try:
                response = requests.get(pts_npy_url)
                response.raise_for_status()
                with open(pts_npy, 'wb') as f:
                    f.write(response.content)
                st.success(f"Downloaded {pts_npy}")
            except Exception as e:
                st.error(f"Error downloading pts_in_hull.npy: {e}")
                # Fallback to local creation
                create_pts_in_hull(pts_npy)
    
    return prototxt, caffemodel, pts_npy

# Fallback function to create prototxt file locally if download fails
def create_prototxt_file(filename):
    st.warning("Creating prototxt file locally as download failed.")
    prototxt_content = """
name: "Colorization"
input: "data_l"
input_dim: 1
input_dim: 1
input_dim: 224
input_dim: 224
layer { name: "conv1_1" type: "Convolution" bottom: "data_l" top: "conv1_1" convolution_param { num_output: 64 pad: 1 kernel_size: 3 } }
layer { name: "relu1_1" type: "ReLU" bottom: "conv1_1" top: "conv1_1" }
layer { name: "conv1_2" type: "Convolution" bottom: "conv1_1" top: "conv1_2" convolution_param { num_output: 64 pad: 1 kernel_size: 3 stride: 2 } }
layer { name: "relu1_2" type: "ReLU" bottom: "conv1_2" top: "conv1_2" }
layer { name: "conv2_1" type: "Convolution" bottom: "conv1_2" top: "conv2_1" convolution_param { num_output: 128 pad: 1 kernel_size: 3 } }
layer { name: "relu2_1" type: "ReLU" bottom: "conv2_1" top: "conv2_1" }
layer { name: "conv2_2" type: "Convolution" bottom: "conv2_1" top: "conv2_2" convolution_param { num_output: 128 pad: 1 kernel_size: 3 stride: 2 } }
layer { name: "relu2_2" type: "ReLU" bottom: "conv2_2" top: "conv2_2" }
layer { name: "conv3_1" type: "Convolution" bottom: "conv2_2" top: "conv3_1" convolution_param { num_output: 256 pad: 1 kernel_size: 3 } }
layer { name: "relu3_1" type: "ReLU" bottom: "conv3_1" top: "conv3_1" }
layer { name: "conv3_2" type: "Convolution" bottom: "conv3_1" top: "conv3_2" convolution_param { num_output: 256 pad: 1 kernel_size: 3 } }
layer { name: "relu3_2" type: "ReLU" bottom: "conv3_2" top: "conv3_2" }
layer { name: "conv3_3" type: "Convolution" bottom: "conv3_2" top: "conv3_3" convolution_param { num_output: 256 pad: 1 kernel_size: 3 stride: 2 } }
layer { name: "relu3_3" type: "ReLU" bottom: "conv3_3" top: "conv3_3" }
layer { name: "conv4_1" type: "Convolution" bottom: "conv3_3" top: "conv4_1" convolution_param { num_output: 512 pad: 1 kernel_size: 3 } }
layer { name: "relu4_1" type: "ReLU" bottom: "conv4_1" top: "conv4_1" }
layer { name: "conv4_2" type: "Convolution" bottom: "conv4_1" top: "conv4_2" convolution_param { num_output: 512 pad: 1 kernel_size: 3 } }
layer { name: "relu4_2" type: "ReLU" bottom: "conv4_2" top: "conv4_2" }
layer { name: "conv4_3" type: "Convolution" bottom: "conv4_2" top: "conv4_3" convolution_param { num_output: 512 pad: 1 kernel_size: 3 } }
layer { name: "relu4_3" type: "ReLU" bottom: "conv4_3" top: "conv4_3" }
layer { name: "conv5_1" type: "Convolution" bottom: "conv4_3" top: "conv5_1" convolution_param { num_output: 512 pad: 2 kernel_size: 3 dilation: 2 } }
layer { name: "relu5_1" type: "ReLU" bottom: "conv5_1" top: "conv5_1" }
layer { name: "conv5_2" type: "Convolution" bottom: "conv5_1" top: "conv5_2" convolution_param { num_output: 512 pad: 2 kernel_size: 3 dilation: 2 } }
layer { name: "relu5_2" type: "ReLU" bottom: "conv5_2" top: "conv5_2" }
layer { name: "conv5_3" type: "Convolution" bottom: "conv5_2" top: "conv5_3" convolution_param { num_output: 512 pad: 2 kernel_size: 3 dilation: 2 } }
layer { name: "relu5_3" type: "ReLU" bottom: "conv5_3" top: "conv5_3" }
layer { name: "conv6_1" type: "Convolution" bottom: "conv5_3" top: "conv6_1" convolution_param { num_output: 512 pad: 2 kernel_size: 3 dilation: 2 } }
layer { name: "relu6_1" type: "ReLU" bottom: "conv6_1" top: "conv6_1" }
layer { name: "conv6_2" type: "Convolution" bottom: "conv6_1" top: "conv6_2" convolution_param { num_output: 512 pad: 2 kernel_size: 3 dilation: 2 } }
layer { name: "relu6_2" type: "ReLU" bottom: "conv6_2" top: "conv6_2" }
layer { name: "conv6_3" type: "Convolution" bottom: "conv6_2" top: "conv6_3" convolution_param { num_output: 512 pad: 2 kernel_size: 3 dilation: 2 } }
layer { name: "relu6_3" type: "ReLU" bottom: "conv6_3" top: "conv6_3" }
layer { name: "conv7_1" type: "Convolution" bottom: "conv6_3" top: "conv7_1" convolution_param { num_output: 512 pad: 1 kernel_size: 3 } }
layer { name: "relu7_1" type: "ReLU" bottom: "conv7_1" top: "conv7_1" }
layer { name: "conv7_2" type: "Convolution" bottom: "conv7_1" top: "conv7_2" convolution_param { num_output: 512 pad: 1 kernel_size: 3 } }
layer { name: "relu7_2" type: "ReLU" bottom: "conv7_2" top: "conv7_2" }
layer { name: "conv7_3" type: "Convolution" bottom: "conv7_2" top: "conv7_3" convolution_param { num_output: 512 pad: 1 kernel_size: 3 } }
layer { name: "relu7_3" type: "ReLU" bottom: "conv7_3" top: "conv7_3" }
layer { name: "conv8_1" type: "Convolution" bottom: "conv7_3" top: "conv8_1" convolution_param { num_output: 256 pad: 1 kernel_size: 3 } }
layer { name: "relu8_1" type: "ReLU" bottom: "conv8_1" top: "conv8_1" }
layer { name: "conv8_2" type: "Convolution" bottom: "conv8_1" top: "conv8_2" convolution_param { num_output: 256 pad: 1 kernel_size: 3 } }
layer { name: "relu8_2" type: "ReLU" bottom: "conv8_2" top: "conv8_2" }
layer { name: "conv8_3" type: "Convolution" bottom: "conv8_2" top: "conv8_3" convolution_param { num_output: 256 pad: 1 kernel_size: 3 } }
layer { name: "relu8_3" type: "ReLU" bottom: "conv8_3" top: "conv8_3" }
layer { name: "conv8_313" type: "Convolution" bottom: "conv8_3" top: "conv8_313" convolution_param { num_output: 313 pad: 1 kernel_size: 1 } }
layer { name: "conv8_313_rh" type: "Scale" bottom: "conv8_313" top: "conv8_313_rh" scale_param { bias_term: false } }
layer { name: "class8_313_rh" type: "Softmax" bottom: "conv8_313_rh" top: "class8_313_rh" }
layer { name: "class8_ab" type: "Convolution" bottom: "class8_313_rh" top: "class8_ab" convolution_param { num_output: 2 pad: 0 kernel_size: 1 } }
layer { name: "Silence" type: "Silence" bottom: "class8_ab" }
    """
    with open(filename, 'w') as f:
        f.write(prototxt_content)
    st.success(f"Created {filename} locally")

# Fallback function to create pts_in_hull.npy file locally if download fails
def create_pts_in_hull(filename):
    st.warning("Creating pts_in_hull.npy file locally as download failed.")
    # This is an approximation of the original file
    response = requests.get('https://gist.github.com/anonymous/b7fc1ecb9b24ba9e0c2f3ba7fecbcbb0/raw/470fe209af3e6df1c3b0c51d54b78889e3d069ae/pts_in_hull.npy')
    pts_in_hull = np.load(io.BytesIO(response.content))
    np.save(filename, pts_in_hull)
    st.success(f"Created {filename} locally")

# Load the colorization model
@st.cache_resource
def load_model(prototxt, caffemodel, pts_npy):
    if not prototxt or not caffemodel or not pts_npy:
        return None
    
    try:
        # Load the model
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        
        # Load the cluster centers
        kernel = np.load(pts_npy)
        
        # Add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        return net
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to colorize an image
def colorize_image(img, net):
    if net is None:
        st.error("Model not loaded properly. Please check the model files.")
        return None, None
    
    # Scale the image
    scaled = img.astype("float32") / 255.0
    
    # Convert to Lab color space
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Resize for the network
    resized = cv2.resize(lab_img, (224, 224))
    
    # Extract and process L channel
    L = cv2.split(resized)[0]
    L -= 50  # Mean subtraction
    
    # Predict ab channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize ab channels to match original size
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
    
    # Get original L channel
    L = cv2.split(lab_img)[0]
    
    # Join L with predicted ab
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    
    # Convert back to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    
    # Convert to 8-bit
    colorized = (255 * colorized).astype("uint8")
    
    # Return original and colorized images in RGB for Streamlit
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    
    return original_rgb, colorized_rgb

# Function to ensure image is grayscale
def process_input_image(image):
    # Check if image is already grayscale
    if len(image.shape) < 3 or image.shape[2] == 1:
        return image
    
    # If it's color, force conversion to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert back to 3 channels for consistency
    gray_3channels = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_3channels

# Function to enhance colors
def enhance_colors(colorized_img, saturation_factor=1.2):
    # Convert to HSV for easier color manipulation
    hsv_img = cv2.cvtColor(colorized_img, cv2.COLOR_RGB2HSV).astype('float32')
    
    # Enhance saturation
    hsv_img[:, :, 1] = hsv_img[:, :, 1] * saturation_factor
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
    
    # Convert back to RGB
    enhanced_img = cv2.cvtColor(hsv_img.astype('uint8'), cv2.COLOR_HSV2RGB)
    
    return enhanced_img

# Main app function
def main():
    # Download and load the models
    prototxt, caffemodel, pts_npy = download_models()
    net = load_model(prototxt, caffemodel, pts_npy)
    
    if net is None:
        st.error("Failed to load the colorization model. Please check the errors above.")
        st.stop()
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # File uploader
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a black and white image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Create a temporary file to save the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_filename = tmp_file.name
            
            # Read the image
            img = cv2.imread(tmp_filename)
            
            if img is None:
                st.error("Error reading the uploaded image. Please try another file.")
                st.stop()
            
            # Options for image processing
            force_grayscale = st.checkbox("Force grayscale conversion", value=True, 
                                        help="If your image has some color, check this to convert it to grayscale first")
            
            enhance_saturation = st.checkbox("Enhance colors in result", value=True,
                                           help="Apply additional color enhancement to make colors more vibrant")
            
            saturation_factor = st.slider("Color saturation", min_value=0.5, max_value=2.0, value=1.2, step=0.1,
                                        help="Adjust the color intensity of the result")
            
            if force_grayscale:
                img = process_input_image(img)
            
            # Display original image
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            
            # Colorize button
            if st.button("Colorize Image"):
                with st.spinner("Processing..."):
                    # Colorize the image
                    try:
                        original, colorized = colorize_image(img, net)
                        
                        if original is not None and colorized is not None:
                            # Apply color enhancement if selected
                            if enhance_saturation:
                                colorized = enhance_colors(colorized, saturation_factor)
                            
                            # Display result in second column
                            with col2:
                                st.subheader("Colorized Result")
                                st.image(colorized, caption="Colorized Image", use_column_width=True)
                                
                                # Add download button
                                # First save the colorized image to a temporary file
                                colorized_filename = tmp_filename.replace('.jpg', '_colorized.jpg')
                                cv2.imwrite(colorized_filename, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
                                
                                with open(colorized_filename, "rb") as file:
                                    btn = st.download_button(
                                        label="Download Colorized Image",
                                        data=file,
                                        file_name=f"colorized_{uploaded_file.name}",
                                        mime=f"image/{uploaded_file.name.split('.')[-1]}"
                                    )
                                
                                # Add button to show before/after comparison
                                if st.button("Show Side-by-Side Comparison"):
                                    # Create a side-by-side comparison
                                    comparison = np.hstack((cv2.resize(original, (400, 400)), 
                                                           cv2.resize(colorized, (400, 400))))
                                    st.image(comparison, caption="Before and After", width=800)
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
            
            # Clean up the temporary file
            try:
                os.unlink(tmp_filename)
                if 'colorized_filename' in locals():
                    os.unlink(colorized_filename)
            except:
                pass
    
    # If no image is uploaded, show example in second column
    if uploaded_file is None:
        with col2:
            st.subheader("Example Result")
            st.write("Upload an image to see the colorization result here.")
    
    # Add some information about the app
    st.markdown("---")
    st.subheader("How it works")
    st.write("""
    This app uses a deep learning model that was trained to predict color information from grayscale images.
    The model works by analyzing patterns and features in the grayscale image to guess what colors might be present.
    
    The colorization process follows these steps:
    1. Convert the grayscale image to the L*a*b* color space (L = lightness, a/b = color dimensions)
    2. Extract the L channel and feed it to the neural network
    3. The network predicts the a and b color channels
    4. Combine the original L channel with the predicted a/b channels
    5. Convert back to RGB color space
    
    While the results can be impressive, the colorization is an educated guess rather than a recovery of the true colors,
    especially for objects that could plausibly be many different colors.
    """)
    
    # App tips
    st.subheader("Tips for best results")
    st.write("""
    - Images with clear subjects and good contrast tend to colorize better
    - Historical photos with good lighting work particularly well
    - If your image already has some color, use the "Force grayscale" option
    - Adjust the color saturation slider to get more vibrant or subtle colors
    - The model works best on photos of natural scenes, people, and common objects
    - Very old or low-quality photographs may produce less accurate colorization.
    """)

if __name__ == "__main__":
    main()
