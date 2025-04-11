import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_PATH = "maskedspec.png"
INPUT_PATH = "maskedspec2.png"
INPUT_PATH = "140101-003651_TRU_tt_1s.jpg"
OUTPUT_IMAGE = "custom_modules/images/detected_syllables.png"
OUTPUT_PLOT = "custom_modules/images/syllable_peaks.png"

# === Load image ===
img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image at {INPUT_PATH}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert so brighter parts become higher values
gray = cv2.bitwise_not(gray)

# Collapse frequency axis to get intensity over time
intensity_over_time = np.mean(gray, axis=0)  # shape: (width,)

# Smooth signal
smoothed = cv2.GaussianBlur(intensity_over_time.reshape(1, -1), (1, 11), 0).flatten()

# Detect peaks (tune `height` and `distance` as needed)
# peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.5, distance=20)
# Detect peaks with stronger filtering
peaks, _ = find_peaks(
    smoothed,
    height=np.max(smoothed) * 0.9,
    distance=20,              # Increase to avoid close peaks
    prominence=1             # Add to ignore small bumps
)


# Draw detected syllable areas as vertical lines
output = img.copy()
for peak in peaks:
    cv2.line(output, (peak, 0), (peak, output.shape[0]), (0, 255, 0), 2)

# Save results
print(f"Detected syllables: {len(peaks)}")
cv2.imwrite(OUTPUT_IMAGE, output)
print(f"Saved syllable-marked spectrogram: {OUTPUT_IMAGE}")

# Save intensity plot with peaks
plt.figure(figsize=(10, 3))
plt.plot(smoothed, label="Intensity over time")
plt.plot(peaks, smoothed[peaks], "rx", label="Detected syllables")
plt.xlabel("Time (pixels)")
plt.ylabel("Average Intensity")
plt.title("Detected Syllables Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"Saved intensity plot: {OUTPUT_PLOT}")
