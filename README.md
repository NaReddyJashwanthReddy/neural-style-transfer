# Neural Style Transfer (NST) Using PyTorch

## Overview
This project implements Neural Style Transfer (NST) using PyTorch. The model uses a pre-trained VGG19 network to extract features and apply style transformations to an input image.

## Features
- Uses a pre-trained VGG19 model for feature extraction.
- Implements content, style, and total variation loss.
- Optimizes the image using the L-BFGS optimizer.
- Supports image upload via Streamlit UI.
- Displays real-time transformed images.
- Allows downloading the final output image.

## Requirements
Make sure you have the following installed:

- Python 3.7+
- PyTorch
- torchvision
- Streamlit
- Pillow
- numpy

You can install the required dependencies using:
```sh
pip install torch torchvision streamlit pillow numpy
```

## File Structure
```
├── nst.py                  # Main script for Neural Style Transfer
├── utils.py                # Helper functions for loading and processing images
├── Loss.py                 # Defines content, style, and total variation loss functions
├── logger.py               # Logging utility for debugging
├── README.txt              # Project documentation
```

## How to Run
1. Run the following command to start the Streamlit UI:
   ```sh
   streamlit run nst.py
   ```
2. Upload a content image and a style image.
3. Click the "Start" button to begin style transfer.
4. The output image will be displayed in real-time.
5. Download the transformed image using the "Download" button.

## Explanation of Key Components

### 1. **NSTLoss Class (`Loss.py`)**
   - `ContentLoss()`: Computes the mean squared error between content and target features.
   - `StyleLoss()`: Computes the style loss using the Gram matrix.
   - `total_variable_loss()`: Adds regularization by penalizing pixel differences.

### 2. **Feature Extraction (`utils.py`)**
   - Uses a pre-trained VGG19 model to extract style and content features.

### 3. **Image Processing (`utils.py`)**
   - `Load_Image()`: Loads and preprocesses an image.
   - `Denormalize()`: Converts a tensor back to an image.
   - `ImShow()`: Converts an image tensor to a NumPy array for display.

### 4. **Training and Optimization (`nst.py`)**
   - Loads images and extracts features.
   - Defines loss functions and optimizer.
   - Runs multiple iterations to minimize the style transfer loss.
   - Displays the generated image using Streamlit.

## Logging
The script logs progress and loss calculations to track the training process. Logs can be found in `logger.py`.

## Troubleshooting
- **CUDA not available**: Ensure PyTorch is installed with CUDA support.
- **Slow processing**: Reduce `steps` or run on a GPU.
- **Streamlit not found**: Install it using `pip install streamlit`.

## Acknowledgments
- This project is inspired by the original Neural Style Transfer paper by Gatys et al.
- Uses PyTorch's pre-trained VGG19 model.

## Author
Nareddy Jashwanth Reddy



