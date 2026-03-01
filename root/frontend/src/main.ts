import { setupCanvas, getCanvasData, setBrushSize, getPixelData } from "./ui/canvas.js";

// ─── Canvas einrichten ────────────────────────────────────────────────────────

const canvas = document.getElementById("drawing-canvas") as HTMLCanvasElement;
setupCanvas(canvas, 20);

// ─── Pinselgröße ──────────────────────────────────────────────────────────────

const brushSlider = document.getElementById("brush-size") as HTMLInputElement;
const brushLabel = document.getElementById("brush-label") as HTMLSpanElement;

brushSlider?.addEventListener("input", () => {
  const size = Number(brushSlider.value);
  setBrushSize(canvas, size);
  if (brushLabel) brushLabel.textContent = `${size} px`;
});

// ─── Leeren ───────────────────────────────────────────────────────────────────

document.getElementById("clear-btn")!.addEventListener("click", () => {
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
});

// ─── Vorhersage ───────────────────────────────────────────────────────────────

document.getElementById("predict-btn")!.addEventListener("click", async () => {
  const dataUrl = getCanvasData(canvas); // base64-PNG für den Server

  // Optional: Rohe Pixeldaten lokal verfügbar (z. B. für clientseitige Modelle)
  // const pixels = getPixelData(canvas);

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  });

  if (!response.ok) {
    alert("Fehler bei Vorhersage:\n" + response.statusText);
    return;
  }

  const { prediction, probabilities } = await response.json();
  document.getElementById("predicted-digit")!.textContent = prediction;
  renderBarChart(probabilities);
  // TODO: Heatmap
});

// ─── Balkendiagramm ───────────────────────────────────────────────────────────

function renderBarChart(probs: number[]): void {
  const style = getComputedStyle(document.documentElement);
  const colorText = style.getPropertyValue("--text").trim();
  const colorAccent = style.getPropertyValue("--accent").trim();
  const colorBorder = style.getPropertyValue("--border").trim();
  const colorPanel = style.getPropertyValue("--panel").trim();
  const colorOther = style.getPropertyValue("--other").trim();

  const container = document.getElementById("bar-chart")!;
  container.innerHTML = "";

  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = container.clientWidth || 300;
  const cssHeight = 160;

  canvas.style.width = cssWidth + "px";
  canvas.style.height = cssHeight + "px";
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  container.appendChild(canvas);

  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);

  const margin = { top: 10, right: 10, bottom: 20, left: 36 };
  const innerWidth = cssWidth - margin.left - margin.right;
  const innerHeight = cssHeight - margin.top - margin.bottom;
  const barWidth = innerWidth / probs.length;

  const FONT = `10px "Segoe UI", sans-serif`;
  const LABEL_THRESHOLD = 18;
  const DURATION = 500; // ms

  function easeOutCubic(t: number): number {
    return 1 - Math.pow(1 - t, 3);
  }

  function draw(progress: number): void {
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    // Hintergrund
    ctx.fillStyle = colorPanel;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    // Y-Achse: Beschriftungen & Tick-Marks alle 20 %
    ctx.font = FONT;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let tick = 0; tick <= 100; tick += 20) {
      const y = margin.top + innerHeight - (tick / 100) * innerHeight;

      ctx.fillStyle = colorText;
      ctx.fillText(`${tick}%`, margin.left - 6, y);

      ctx.strokeStyle = colorBorder;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.left - 3, y);
      ctx.lineTo(margin.left, y);
      ctx.stroke();
    }

    // Achsenlinien
    ctx.strokeStyle = colorBorder;
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + innerHeight);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + innerHeight);
    ctx.lineTo(margin.left + innerWidth, margin.top + innerHeight);
    ctx.stroke();

    // Balken + Labels
    probs.forEach((prob, i) => {
      const animatedHeight = prob * progress * innerHeight;
      const x = margin.left + i * barWidth;
      const barTop = margin.top + innerHeight - animatedHeight;

      ctx.fillStyle = colorAccent;
      ctx.fillRect(x + 1, barTop, barWidth - 3, animatedHeight);

      // X-Label (Ziffer)
      ctx.font = FONT;
      ctx.fillStyle = colorText;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(i.toString(), x + barWidth / 2, margin.top + innerHeight + 5);

      // Prozent-Beschriftung (blendet sich am Ende der Animation ein)
      if (prob > 0 && progress > 0.8) {
        const labelOpacity = (progress - 0.8) / 0.2;
        const label = `${Math.round(prob * 100)}%`;
        const inside = prob * innerHeight >= LABEL_THRESHOLD;

        ctx.font = `bold ${FONT}`;
        ctx.textAlign = "center";
        ctx.globalAlpha = labelOpacity;

        if (inside) {
          ctx.fillStyle = colorOther;
          ctx.textBaseline = "bottom";
          ctx.fillText(label, x + barWidth / 2, barTop + animatedHeight - 3);
        } else {
          ctx.fillStyle = colorText;
          ctx.textBaseline = "bottom";
          ctx.fillText(label, x + barWidth / 2, barTop - 2);
        }

        ctx.globalAlpha = 1;
      }
    });
  }

  const start = performance.now();

  function animate(now: number): void {
    const elapsed = now - start;
    const progress = easeOutCubic(Math.min(elapsed / DURATION, 1));
    draw(progress);
    if (elapsed < DURATION) requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}
