import os
from PIL import Image, ImageFilter
import cv2
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator

input_folder = 'data'
output_folder = 'output'

os.makedirs(output_folder, exist_ok=True)


def determine_clusters(image, min_clusters=2, max_clusters=15):
    image_flat = image.flatten().astype(np.float32)
    distortions = []
    K = range(min_clusters, max_clusters+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(image_flat.reshape(-1, 1))
        distortions.append(kmeans.inertia_)
    kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    optimal_k = kn.knee if kn.knee else max_clusters
    return optimal_k

def process_and_segment_images(blur_radius=30, scale=0.5):
    for file_name in os.listdir(input_folder):
        if file_name.startswith("tm") and file_name.lower().endswith(('.png')):
            input_path = os.path.join(input_folder, file_name)
            
            with Image.open(input_path) as img:
                blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                blurred_img = np.array(blurred_img)
                
                if len(blurred_img.shape) == 3:
                    blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
            
            scaled_image = cv2.resize(blurred_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            clusters = determine_clusters(scaled_image)

            segmented = cv2.kmeans(
                scaled_image.flatten().astype(np.float32),
                clusters,
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                10,
                cv2.KMEANS_RANDOM_CENTERS
            )[1].reshape(scaled_image.shape)

            segmented_full_size = cv2.resize(segmented, (blurred_img.shape[1], blurred_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            output_name = file_name.replace("tm", "seg")
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, segmented_full_size)

            print(f"Processed: {file_name}")

if __name__ == "__main__":
    process_and_segment_images()