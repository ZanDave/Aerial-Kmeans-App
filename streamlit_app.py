import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from pathlib import Path
import pickle
import time
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2lab, lab2rgb
from scipy.stats import mode

class EnhancedImageClusterer:
    def __init__(self):
        self.kmeans = None
        self.model_path = "enhanced_trained_model.pkl"

    @staticmethod
    def preprocess_image(image, max_size=800):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)

        return np.array(image)

    def extract_features(self, image):
        lab_image = rgb2lab(image)
        features = lab_image.reshape(-1, 3)
        return features

    def train_model(self, image_paths, progress_callback):
        features_list = []
        total_images = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
                image_array = self.preprocess_image(image)
                features = self.extract_features(image_array)
                features_list.append(features)
            except Exception as e:
                st.warning(f"Skipping image {img_path}: {str(e)}")

            progress = (idx + 1) / total_images
            progress_callback(progress)

        if not features_list:
            raise ValueError("No valid images found for training")

        all_features = np.vstack(features_list)
        
        # Use MiniBatchKMeans instead of KMeans to reduce memory usage
        self.kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=1000)
        self.kmeans.fit(all_features)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.kmeans, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.kmeans = pickle.load(f)
            return True
        return False

    def cluster_image(self, image, n_clusters, progress_callback):
        # Felzenszwalb segmentation
        segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

        # Extract features for each segment
        features = []
        for i in range(np.max(segments) + 1):
            mask = segments == i
            segment = image[mask]
            if len(segment) > 0:
                avg_color = np.mean(segment, axis=0)
                features.append(avg_color)

        features = np.array(features)

        # Convert to LAB color space
        lab_features = rgb2lab(features.reshape(1, -1, 3)).reshape(-1, 3)

        # Perform MiniBatchKMeans clustering
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(lab_features)

        # Map labels back to pixels
        pixel_labels = np.zeros(segments.shape, dtype=np.int32)
        for i, label in enumerate(labels):
            pixel_labels[segments == i] = label

        # Calculate silhouette score
        silhouette_avg = silhouette_score(lab_features, labels)

        return pixel_labels, silhouette_avg

def create_visualization(image, labels, n_clusters, silhouette_avg):
    # Create a color map
    color_map = plt.get_cmap('tab10')
    rgba_colors = color_map(np.linspace(0, 1, n_clusters))
    rgb_colors = rgba_colors[:, :3]

    # Create clustered image
    clustered_img = np.zeros_like(image)
    for i in range(n_clusters):
        mask = labels == i
        clustered_img[mask] = (rgb_colors[i] * 255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(clustered_img)
    ax2.set_title(f"Clustered Image (Silhouette Score: {silhouette_avg:.2f})")
    ax2.axis('off')

    # Add cluster numbers
    for i in range(n_clusters):
        mask = labels == i
        if mask.sum() > 0:
            y_coords, x_coords = np.where(mask)
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            ax2.text(centroid_x, centroid_y, str(i+1), color='white', fontsize=12,
                     fontweight='bold', ha='center', va='center',
                     bbox=dict(facecolor='black', edgecolor='none', alpha=0.7, pad=2))

    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="Enhanced Aerial Image Clustering", layout="wide")

    st.title("Enhanced Aerial Image Clustering")

    # Use tabs for better organization
    tab1, tab2 = st.tabs(["Model Training", "Image Clustering"])

    with tab1:
        st.header("Model Training")
        
        model_option = st.radio(
            "Choose Model Source",
            ["Train New Model", "Load Existing Model"],
            help="Select whether to train a new model or load a pre-trained one"
        )
        
        if model_option == "Train New Model":
            training_folder = st.text_input(
                "Training Images Folder Path",
                value="public/training_data",
                help="Path to folder containing training images"
            )
            
            if os.path.exists(training_folder):
                image_files = list(Path(training_folder).glob("*.jpg")) + \
                              list(Path(training_folder).glob("*.png"))
                st.success(f"Found {len(image_files)} training images")
                
                if len(image_files) >= 5:
                    if st.button("Train Model", key="train_model"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        clusterer = EnhancedImageClusterer()
                        try:
                            clusterer.train_model(image_files, progress_bar.progress)
                            status_text.success("Model trained successfully!")
                        except Exception as e:
                            status_text.error(f"Error during model training: {str(e)}")
                else:
                    st.error("Insufficient training data (need at least 5 images)")
            else:
                st.error("Training folder not found")
        
        elif model_option == "Load Existing Model":
            clusterer = EnhancedImageClusterer()
            if clusterer.load_model():
                st.success("Model loaded successfully")
            else:
                st.error("No pre-trained model found")

    with tab2:
        st.header("Image Clustering")
        
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=3,
            help="Choose the number of regions to identify"
        )

        uploaded_file = st.file_uploader("Upload an aerial image for clustering", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = EnhancedImageClusterer.preprocess_image(image)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Perform Clustering", key="perform_clustering"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                start_time = time.time()
                clusterer = EnhancedImageClusterer()
                labels, silhouette_avg = clusterer.cluster_image(image_array, n_clusters, progress_bar.progress)
                end_time = time.time()

                processing_time = end_time - start_time
                status_text.success(f"Clustering completed in {processing_time:.2f} seconds")

                fig = create_visualization(image_array, labels, n_clusters, silhouette_avg)
                st.pyplot(fig)

                st.markdown("### Clustering Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Clusters", n_clusters)
                with col2:
                    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")

                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                st.download_button(
                    label="Download Clustered Image",
                    data=buf.getvalue(),
                    file_name="clustered_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()