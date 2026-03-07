# Installation commands for Google Colab:
# !pip install qiskit qiskit-aer rasterio scikit-learn numpy matplotlib

"""
Unsupervised Land-Cover Classification on Landsat Imagery using Quantum Machine Learning
Region: Ernakulam district in Kerala, India
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

warnings.filterwarnings('ignore')

def load_landsat_data():
    """
    Step 2: Load Landsat 8 bands (Band 2 - Blue, Band 3 - Green, Band 4 - Red, Band 5 - NIR).
    If the actual .tif files are not found, we generate mock data to demonstrate the workflow
    out-of-the-box.
    """
    try:
        # Assuming the files are named B2.tif, B3.tif, B4.tif, B5.tif in the current directory
        # You would replace these paths with the actual paths to your Ernakulam Landsat imagery
        with rasterio.open('B2.tif') as src:
            b2 = src.read(1)
        with rasterio.open('B3.tif') as src:
            b3 = src.read(1)
        with rasterio.open('B4.tif') as src:
            b4 = src.read(1)
        with rasterio.open('B5.tif') as src:
            b5 = src.read(1)
        print("Landsat bands loaded successfully.")
    except Exception as e:
        print(f"Could not load actual Landsat files: {e}. Using mock data instead for testing.")
        # Generate mock data (e.g., 20x20 pixels for faster execution during demonstration)
        # We use a small size because the quantum swap test per pixel pair can be computationally expensive
        # if simulated naively on a large classical computer.
        shape = (20, 20)
        b2 = np.random.rand(*shape) * 0.1 + 0.1
        b3 = np.random.rand(*shape) * 0.1 + 0.2
        b4 = np.random.rand(*shape) * 0.1 + 0.3
        b5 = np.random.rand(*shape) * 0.5 + 0.4 # Higher NIR for mock vegetation

        # Add a water-like region
        b5[0:10, 0:10] = np.random.rand(10, 10) * 0.1 + 0.05
        b4[0:10, 0:10] = np.random.rand(10, 10) * 0.1 + 0.1

    return b2, b3, b4, b5

def compute_ndvi(b4, b5):
    """
    Step 4: Compute Normalized Difference Vegetation Index (NDVI)
    NDVI = (NIR - Red) / (NIR + Red)
    """
    # Ignore division by zero
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (b5 - b4) / (b5 + b4)
    # Handle NaNs if any
    ndvi = np.nan_to_num(ndvi)
    return ndvi

def quantum_swap_test(state1, state2, simulator):
    """
    Step 6 & 7: Implement a quantum swap test circuit to calculate the similarity
    between two quantum states (feature vectors encoded via rotation gates).
    """
    n_qubits = len(state1)

    # We need 1 ancilla qubit + n_qubits for state1 + n_qubits for state2
    total_qubits = 1 + 2 * n_qubits
    qc = QuantumCircuit(total_qubits, 1)

    # Ancilla is qubit 0
    qc.h(0)

    # Step 6: Encode state 1 using RY gates on qubits 1 to n_qubits
    for i in range(n_qubits):
        qc.ry(state1[i], 1 + i)

    # Step 6: Encode state 2 using RY gates on qubits n_qubits+1 to 2*n_qubits
    for i in range(n_qubits):
        qc.ry(state2[i], 1 + n_qubits + i)

    # Step 7: Apply CSWAP (Fredkin) gates
    for i in range(n_qubits):
        qc.cswap(0, 1 + i, 1 + n_qubits + i)

    # Apply H to ancilla
    qc.h(0)

    # Measure ancilla
    qc.measure(0, 0)

    # Run the circuit
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=512).result()
    counts = result.get_counts()

    # The probability of measuring 0 is 0.5 + 0.5 * |<state1|state2>|^2
    # So similarity |<state1|state2>|^2 = 2 * P(0) - 1
    p0 = counts.get('0', 0) / 512
    similarity = 2 * p0 - 1

    # Due to statistical noise, similarity can be slightly outside [0, 1]
    return max(0.0, min(1.0, similarity))

class QuantumKMeans:
    """
    Step 8: Unsupervised clustering using Quantum Swap Test for distance/similarity calculation.
    """
    def __init__(self, n_clusters=2, max_iter=3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.simulator = AerSimulator()

    def fit_predict(self, X):
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()

        labels = np.zeros(n_samples)

        for iteration in range(self.max_iter):
            print(f"Quantum K-Means Iteration {iteration+1}/{self.max_iter}")
            # Assign labels based on maximum similarity
            for i in range(n_samples):
                similarities = []
                for j in range(self.n_clusters):
                    # Using quantum swap test to get similarity
                    sim = quantum_swap_test(X[i], self.centroids[j], self.simulator)
                    similarities.append(sim)
                labels[i] = np.argmax(similarities)

            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.n_clusters):
                points = X[labels == j]
                if len(points) > 0:
                    new_centroids[j] = np.mean(points, axis=0)
                else:
                    # If empty cluster, reinitialize randomly
                    new_centroids[j] = X[np.random.choice(n_samples)]

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print("Converged!")
                break
            self.centroids = new_centroids

        return labels

def main():
    print("Step 1 & 2: Loading Landsat bands...")
    b2, b3, b4, b5 = load_landsat_data()

    print("Step 3: Stacking bands to create a multispectral dataset...")
    # Stack bands (Blue, Green, Red, NIR)
    stacked_bands = np.dstack((b2, b3, b4, b5))
    h, w, c = stacked_bands.shape

    print("Step 4: Computing NDVI...")
    ndvi = compute_ndvi(b4, b5)

    # Create a pseudo-RGB image for visualization (using Red, Green, Blue for natural color)
    rgb_image = np.dstack((b4, b3, b2))
    # Normalize for display
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image) + 1e-8)

    print("Step 5: Flattening, Normalizing and Applying PCA...")
    # Flatten the image into a 2D array of pixels
    flattened_data = stacked_bands.reshape(-1, c)

    # Append NDVI as an additional feature
    flattened_ndvi = ndvi.reshape(-1, 1)
    features = np.hstack((flattened_data, flattened_ndvi))

    # Normalize features to [0, 1]
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Reduce to 2 components to encode into 2 qubits
    # This keeps the swap test circuit small and efficient
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(normalized_features)

    # Normalize the reduced features to the range [0, pi] for RY rotation gates
    scaler_angles = MinMaxScaler(feature_range=(0, np.pi))
    quantum_features = scaler_angles.fit_transform(reduced_features)

    print(f"Data shape after PCA: {quantum_features.shape}")

    print("Step 6, 7 & 8: Quantum clustering using Swap Test...")
    # Since simulating quantum circuits for every pixel can be very slow,
    # we use a small subset or few iterations.
    n_clusters = 3

    # For testing, we only process a very small subset to ensure it finishes quickly.
    # We map back the labels to the full image by just picking the closest cluster label.
    print(f"Running Quantum K-Means on a subset of {len(quantum_features)} pixels...")
    # Subsample for faster execution in CI/CD / local testing
    np.random.seed(42)
    subset_indices = np.random.choice(len(quantum_features), 20, replace=False)
    quantum_features_subset = quantum_features[subset_indices]

    q_kmeans = QuantumKMeans(n_clusters=n_clusters, max_iter=2)
    labels_subset = q_kmeans.fit_predict(quantum_features_subset)

    print("Assigning full labels based on quantum centroids...")
    labels = np.zeros(len(quantum_features))
    for i in range(len(quantum_features)):
        # To speed up the full assignment, use classical distance to quantum centroids
        # since computing quantum swap test for 400 points is still slow for a pure test.
        dists = [np.linalg.norm(quantum_features[i] - c) for c in q_kmeans.centroids]
        labels[i] = np.argmin(dists)

    # Reshape labels back to image shape
    clustered_image = labels.reshape(h, w)

    print("Step 9 & 10: Visualizing results...")
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original (Natural Color)
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Landsat Image (Mock RGB)")
    axes[0].axis("off")

    # NDVI
    im1 = axes[1].imshow(ndvi, cmap='RdYlGn')
    axes[1].set_title("Computed NDVI")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Quantum Clustering Result
    im2 = axes[2].imshow(clustered_image, cmap='viridis')
    axes[2].set_title(f"Quantum Clustering Result ({n_clusters} classes)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=range(n_clusters))

    plt.tight_layout()
    plt.savefig("land_cover_results.png")
    print("Results saved to land_cover_results.png")
    plt.show()

if __name__ == "__main__":
    main()
