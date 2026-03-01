// ui/canvas.ts

/** Intern gespeicherter Zustand pro Canvas-Element */
interface CanvasState {
  brushSize: number;
}

const stateMap = new WeakMap<HTMLCanvasElement, CanvasState>();

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Richtet das Canvas zum Freihand-Zeichnen ein.
 * - Nur schwarz auf weißem Hintergrund
 * - Weicher Pinsel: in der Mitte volle Deckkraft, zum Rand hin transparent (Radial-Gradient)
 * - Keine Lücken bei schnellen Bewegungen (Zwischen-Dots entlang der Strecke)
 */
export function setupCanvas(canvas: HTMLCanvasElement, initialBrushSize = 20): void {
  const ctx = canvas.getContext("2d")!;

  // Weißer Hintergrund
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  stateMap.set(canvas, { brushSize: initialBrushSize });

  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  // ── Zeichenhilfen ──────────────────────────────────────────────────────────

  function drawSoftDot(x: number, y: number): void {
    const state = stateMap.get(canvas)!;
    const r = state.brushSize / 2;

    // Radialer Gradient: Kern schwarz (opak) → Rand transparent
    const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0.0, "rgba(0, 0, 0, 1.0)");
    grad.addColorStop(0.55, "rgba(0, 0, 0, 0.85)");
    grad.addColorStop(0.85, "rgba(0, 0, 0, 0.25)");
    grad.addColorStop(1.0, "rgba(0, 0, 0, 0.0)");

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
  }

  /**
   * Interpoliert zwischen zwei Punkten und setzt Dots,
   * damit auch bei schnellen Bewegungen keine Lücken entstehen.
   */
  function drawSoftLine(x1: number, y1: number, x2: number, y2: number): void {
    const state = stateMap.get(canvas)!;
    const dist = Math.hypot(x2 - x1, y2 - y1);
    // Schrittweite: 20 % des Pinselradius → dichte Abdeckung ohne zu viele Draws
    const step = Math.max(1, state.brushSize * 0.1);
    const steps = Math.ceil(dist / step);

    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      drawSoftDot(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t);
    }
  }

  // ── Koordinaten (DPR-aware) ───────────────────────────────────────────────

  function getPos(e: PointerEvent): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  // ── Event-Listener ────────────────────────────────────────────────────────

  canvas.addEventListener("pointerdown", (e) => {
    isDrawing = true;
    const { x, y } = getPos(e);
    lastX = x;
    lastY = y;
    drawSoftDot(x, y);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!isDrawing) return;
    const { x, y } = getPos(e);
    drawSoftLine(lastX, lastY, x, y);
    lastX = x;
    lastY = y;
  });

  const stopDrawing = () => {
    isDrawing = false;
  };
  canvas.addEventListener("pointerup", stopDrawing);
  canvas.addEventListener("pointerleave", stopDrawing);

  canvas.style.cursor = "crosshair";
  canvas.style.touchAction = "none";
}

// ─────────────────────────────────────────────────────────────────────────────

/** Ändert die Pinselgröße (Durchmesser in Pixel) eines bereits eingerichteten Canvas. */
export function setBrushSize(canvas: HTMLCanvasElement, size: number): void {
  const state = stateMap.get(canvas);
  if (!state) throw new Error("Canvas wurde noch nicht mit setupCanvas() initialisiert.");
  state.brushSize = Math.max(1, size);
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Gibt den Canvas-Inhalt als Base64-kodiertes PNG zurück.
 * Wird direkt an den /predict-Endpunkt geschickt.
 */
export function getCanvasData(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL("image/png");
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Gibt die rohen RGBA-Pixeldaten des Canvas zurück.
 * Jeder Pixel = 4 Werte [R, G, B, A], zeilenweise geordnet.
 * Die Weiterverarbeitung (Graustufen, Normalisierung, Resize) übernimmt der Server.
 */
export function getPixelData(canvas: HTMLCanvasElement): number[] {
  const ctx = canvas.getContext("2d")!;
  const { width, height } = canvas;
  return Array.from(ctx.getImageData(0, 0, width, height).data);
}
