import cv2
import numpy as np
import matplotlib.pyplot as plt

class VCImagePreprocessor:
    """
    Image Preprocessing for Size-Invariant Visual Cryptography
    Steps:
    1. Edge-Preserving Smoothing (Bilateral Filter)
    2. AHP Normalization
    3. CLAHE + Gamma Correction
    4. Canny Edge Detection
    5. Edge-Aware Contrast Map Generation
    """
    def __init__(self, clahe_clip=2.0, gamma=1.2):
        self.clahe_clip = clahe_clip
        self.gamma = gamma
    
    # ---------- STEP 1: EDGE-PRESERVING FILTER ----------
    def guided_filter(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.bilateralFilter(
            image,
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )
    
    # ---------- STEP 2: AHP NORMALIZATION ----------
    def ahp_normalization(self, image):
        min_val = np.percentile(image, 2)
        max_val = np.percentile(image, 98)
        normalized = np.clip(
            (image - min_val) * 255.0 / (max_val - min_val + 1e-6),
            0, 255
        )
        return normalized.astype(np.uint8)
    
    # ---------- STEP 3: CLAHE + GAMMA ----------
    def clahe_gamma_correction(self, image):
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(8, 8)
        )
        clahe_img = clahe.apply(image)
        gamma_corrected = np.array(
            255 * (clahe_img / 255.0) ** (1 / self.gamma),
            dtype=np.uint8
        )
        return gamma_corrected
    
    # ---------- STEP 4: CANNY EDGE ----------
    def canny_edge_detection(self, image, low=50, high=150):
        return cv2.Canny(image, low, high)
    
    # ---------- STEP 5: EDGE-AWARE CONTRAST MAP ----------
    def edge_aware_contrast_map(self, image, edges):
        edge_norm = edges.astype(np.float32) / 255.0
        edge_blur = cv2.GaussianBlur(edge_norm, (5, 5), 1.0)
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_norm = cv2.normalize(
            gradient_mag, None, 0, 1, cv2.NORM_MINMAX
        )
        contrast_map = 0.7 * edge_blur + 0.3 * gradient_norm
        return np.clip(contrast_map, 0, 1)
    
    # ---------- FULL PIPELINE ----------
    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("❌ Image not found!")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = self.guided_filter(gray)
        normalized = self.ahp_normalization(filtered)
        enhanced = self.clahe_gamma_correction(normalized)
        edges = self.canny_edge_detection(enhanced)
        contrast_map = self.edge_aware_contrast_map(enhanced, edges)
        
        return {
            "original": gray,
            "filtered": filtered,
            "normalized": normalized,
            "enhanced": enhanced,
            "edges": edges,
            "contrast_map": contrast_map
        }
    
    # ---------- VISUALIZATION ----------
    def visualize_results(self, results):
        titles = [
            "Original",
            "Edge-Preserving Filter",
            "AHP Normalized",
            "CLAHE + Gamma",
            "Canny Edges",
            "Contrast Map"
        ]
        images = [
            results["original"],
            results["filtered"],
            results["normalized"],
            results["enhanced"],
            results["edges"],
            results["contrast_map"]
        ]
        
        plt.figure(figsize=(15, 10))
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            if i == 5:
                plt.imshow(images[i], cmap="hot", vmin=0, vmax=1)
                plt.colorbar()
            else:
                plt.imshow(images[i], cmap="gray")
            plt.title(titles[i])
            plt.axis("off")
        plt.tight_layout()
        plt.show()

# ===================== MAIN =====================
if __name__ == "__main__":
    preprocessor = VCImagePreprocessor(
        clahe_clip=2.0,
        gamma=1.2
    )
    image_path = "secret_image.png"
    results = preprocessor.preprocess(image_path)
    
    # ✅ SAVE FIRST
    cv2.imwrite("enhanced_image.png", results["enhanced"])
    cv2.imwrite(
        "contrast_map.png",
        (results["contrast_map"] * 255).astype(np.uint8)
    )
    print("✅ Preprocessing completed successfully")
    
    # 🖼️ SHOW LAST
    preprocessor.visualize_results(results)