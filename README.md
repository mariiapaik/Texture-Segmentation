# 📷 Image Segmentation using K-Means Clustering

## 🚀 Overview

This Python script automates image segmentation using **K-Means clustering**, optimized through the elbow method to determine the optimal number of clusters. It's designed to efficiently process multiple images by applying Gaussian blur and resizing before segmenting.

---

## ⚙️ How It Works

1. **Input Processing**
   - Reads all `.png` images starting with `tm` from the `data` directory.

2. **Image Preprocessing**
   - Applies a Gaussian blur (radius configurable, default = 30) to smooth images.
   - Converts images to grayscale if necessary.
   - Resizes images (default scaling factor = 0.5) for efficient processing.

3. **Optimal Cluster Determination**
   - Uses the elbow method (via `KneeLocator`) to automatically determine the optimal number of clusters for K-Means clustering.

4. **Segmentation**
   - Applies K-Means clustering using the determined optimal number of clusters.
   - Resizes the segmented images back to the original dimensions.

5. **Output**
   - Saves segmented images to the `output` directory, renaming them with the prefix `seg`.

---

## 📂 Directory Structure

- **Input folder**: `data/`
  - Images named like `tm*.png`.
- **Output folder**: `output/`
  - Segmented images saved as `seg*.png`.
