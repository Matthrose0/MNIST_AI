from flask import Flask, request, jsonify, send_from_directory
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Any
import os
from numpy.typing import NDArray
from scipy.ndimage import center_of_mass, shift, gaussian_filter

from network.core import NetworkCore

# preparing network
network = NetworkCore.create_network_from_file(28 * 28, "MNIST/network0.csv")


# frontend/public als statischer Ordner
app = Flask(__name__, static_folder="../frontend/public", static_url_path="")

def crop_and_resize(arr: NDArray[np.float32], target: int = 28, padding: int = 2) -> NDArray[np.float32]:
    rows = np.any(arr > 0.05, axis=1)
    cols = np.any(arr > 0.05, axis=0)
    if not rows.any():
        return arr
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Padding hinzufügen wie in MNIST
    rmin = max(0, rmin - padding)
    rmax = min(arr.shape[0] - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(arr.shape[1] - 1, cmax + padding)
    cropped = arr[rmin:rmax+1, cmin:cmax+1]
    img = Image.fromarray((cropped * 255).astype(np.uint8))
    img = img.resize((target, target), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def center_image(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Verschiebt das Bild so, dass der Schwerpunkt in der Mitte liegt."""
    cy, cx = center_of_mass(arr)
    rows, cols = arr.shape
    shift_y = rows / 2 - cy # type: ignore
    shift_x = cols / 2 - cx # type: ignore
    return shift(arr, [shift_y, shift_x], cval=0.0) # type: ignore


@app.route("/")
def index() -> Any:
    assert app.static_folder is not None
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Kein Bild übergeben."}), 400

        # ── Bild dekodieren ───────────────────────────────────────────────────
        base64_data = data["image"].split(",", 1)[1]
        image_bytes = base64.b64decode(base64_data)

        with Image.open(BytesIO(image_bytes)) as img:
            # RGBA → Graustufen → 28×28
            gray = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)

        # Numpy-Array: uint8 [0..255], weiß = 255, schwarz = 0
        arr = np.array(gray, dtype=np.float32)


        # Invertieren (MNIST-Konvention: 0 = Hintergrund, 1 = Strich)
        normalized = (255.0 - arr) / 255.0   # shape (28, 28)

        # ── Vorverarbeitung ───────────────────────────────────────────────────
        normalized = crop_and_resize(normalized)             # eng um den Strich croppen
        normalized = center_image(normalized)                # Schwerpunkt in die Mitte
        normalized = gaussian_filter(normalized, sigma=0.5) # leicht glätten # type: ignore
        normalized = np.clip(normalized, 0.0, 1.0)          # Werte im gültigen Bereich halten  # type: ignore

        # flat = normalized.flatten().reshape(1, -1)           # shape (1, 784)
        flat = normalized.flatten()          # shape (784,)

        # ── Image output for debug ────────────────────────────────────────────
        # print(normalized)
        # import matplotlib.pyplot as plt
        # plt.imshow(arr, cmap="gray") # type: ignore
        # plt.show() # type: ignore
        
        # ── Vorhersage ────────────────────────────────────────────────────────
        output = NetworkCore.forward_pass(network, flat.reshape(1, -1))
        probabilities: list[float] = output.flatten().astype(float).tolist()
        prediction = int(np.argmax(probabilities))

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()   # vollständiger Stack-Trace in der Konsole
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)