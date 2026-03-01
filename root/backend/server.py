from flask import Flask, request, jsonify, send_from_directory
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Any
# from scipy.ndimage import center_of_mass, shift

from network.core import NetworkCore

# preparing network
network = NetworkCore.create_network_from_file(28 * 28, "MNIST/network0.csv")


# frontend/public als statischer Ordner
app = Flask(__name__, static_folder="../frontend/public", static_url_path="")

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
        # und auf [0.0, 1.0] normalisieren
        normalized = (255.0 - arr) / 255.0   # shape (28, 28)
        flat = normalized.flatten()          # shape (784,)

        # ── Image output for debug ────────────────────────────────────────────
        print(normalized)
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
    app.run(debug=True)