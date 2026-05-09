import { createServerFn } from "@tanstack/react-start";

/**
 * Predict server function.
 *
 * In production, point BACKEND_URL to your FastAPI/Flask service that loads
 * the real PyTorch (.pth) or TensorFlow (.h5) model and returns:
 *   { prediction: "PNEUMONIA" | "NORMAL", confidence: number }
 *
 * If BACKEND_URL is not set, this returns a deterministic mock so the UI
 * remains demo-able. The real backend code lives in /backend (FastAPI).
 */
export const predictXray = createServerFn({ method: "POST" })
  .inputValidator((input: { imageBase64: string; filename: string }) => input)
  .handler(async ({ data }) => {
    const backend = process.env.BACKEND_URL;

    if (backend) {
      try {
        const bin = Uint8Array.from(atob(data.imageBase64), (c) => c.charCodeAt(0));
        const form = new FormData();
        form.append("file", new Blob([bin]), data.filename);
        const res = await fetch(`${backend}/predict`, { method: "POST", body: form });
        if (!res.ok) throw new Error(`Backend ${res.status}`);
        const json = (await res.json()) as { prediction: string; confidence: number };
        return { prediction: json.prediction, confidence: json.confidence };
      } catch (err) {
        console.error("Backend predict failed:", err);
        return { prediction: "ERROR", confidence: 0, error: String(err) };
      }
    }

    // Deterministic demo mock based on filename hash
    let h = 0;
    for (let i = 0; i < data.filename.length; i++) h = (h * 31 + data.filename.charCodeAt(i)) | 0;
    const isPneumonia = Math.abs(h) % 100 > 45;
    const confidence = 0.78 + (Math.abs(h) % 22) / 100;
    await new Promise((r) => setTimeout(r, 1400));
    return {
      prediction: isPneumonia ? "PNEUMONIA" : "NORMAL",
      confidence,
      mock: true,
    };
  });
