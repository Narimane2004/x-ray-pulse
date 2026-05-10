import { createFileRoute } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useCallback, useRef, useState } from "react";
import { Lung3D } from "@/components/Lung3D";
import { predictXray } from "@/lib/predict.functions";
import { Button } from "@/components/ui/button";
import { Activity, Upload, Sparkles, ShieldCheck, AlertTriangle } from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "PneumoScan — Détection de pneumonie sur radiographie thoracique" },
      {
        name: "description",
        content:
          "Importez une radiographie thoracique et obtenez une analyse (Normal / Pneumonie) avec score de confiance. Outil d'aide au diagnostic.",
      },
      { name: "keywords", content: "pneumonie, radiographie thoracique, x-ray, détection pneumonie, imagerie médicale, diagnostic pulmonaire" },
      { property: "og:title", content: "PneumoScan — Détection de pneumonie sur radiographie" },
      { property: "og:description", content: "Analyse de radiographie thoracique avec score de confiance." },
      { property: "og:type", content: "website" },
      { name: "twitter:title", content: "PneumoScan — Détection de pneumonie" },
      { name: "twitter:description", content: "Analyse de radiographie thoracique avec score de confiance." },
    ],
    links: [{ rel: "canonical", href: "/" }],
    scripts: [
      {
        type: "application/ld+json",
        children: JSON.stringify({
          "@context": "https://schema.org",
          "@type": "MedicalWebPage",
          name: "PneumoScan",
          description:
            "Outil d'aide au diagnostic permettant l'analyse de radiographies thoraciques pour la détection de la pneumonie.",
          inLanguage: "fr",
          about: { "@type": "MedicalCondition", name: "Pneumonie" },
        }),
      },
    ],
  }),
  component: Index,
});

type Status = "idle" | "loading" | "normal" | "pneumonia";
type HistoryItem = { name: string; prediction: string; confidence: number; at: number };

function Index() {
  const predict = useServerFn(predictXray);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<{ prediction: string; confidence: number; mock?: boolean } | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setStatus("idle");
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setStatus("loading");
    setResult(null);
    const buf = await file.arrayBuffer();
    let bin = "";
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
    const b64 = btoa(bin);
    try {
      const res = await predict({ data: { imageBase64: b64, filename: file.name } });
      const isP = res.prediction === "PNEUMONIA";
      setStatus(isP ? "pneumonia" : "normal");
      setResult(res);
      setHistory((h) => [{ name: file.name, prediction: res.prediction, confidence: res.confidence, at: Date.now() }, ...h].slice(0, 6));
    } catch (e) {
      console.error(e);
      setStatus("idle");
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setStatus("idle");
  };

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Ambient backdrop glow */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full blur-3xl opacity-30"
             style={{ background: "radial-gradient(circle, var(--neon) 0%, transparent 60%)" }} />
      </div>

      <header className="container mx-auto px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl glass glow flex items-center justify-center">
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-glow">PneumoScan</h1>
            <p className="text-xs text-muted-foreground">Medical Imaging</p>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground glass px-3 py-1.5 rounded-full">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          Système prêt
        </div>
      </header>

      <section className="container mx-auto px-6 pt-8 pb-16 grid lg:grid-cols-2 gap-10 items-center">
        {/* LEFT: hero text + upload */}
        <div className="space-y-8">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-primary glass px-3 py-1 rounded-full mb-4">
              <Sparkles className="w-3 h-3" /> Aperçu d'analyse diagnostique
            </div>
            <h2 className="text-4xl md:text-5xl font-bold leading-tight tracking-tight">
              Détection de la <span className="text-primary text-glow">pneumonie</span> par radiographie
            </h2>
            <p className="mt-4 text-muted-foreground max-w-lg">
              Importez une image de radiographie thoracique. La structure pulmonaire est analysée
              et un résultat est fourni avec un score de confiance.
            </p>
          </div>

          {/* Upload zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`glass rounded-2xl p-6 cursor-pointer transition-all relative overflow-hidden ${drag ? "glow scale-[1.01]" : ""}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            {preview ? (
              <div className="relative">
                <img src={preview} alt="X-ray preview" className="w-full max-h-80 object-contain rounded-xl" />
                {status === "loading" && (
                  <div className="absolute inset-0 overflow-hidden rounded-xl">
                    <div className="absolute inset-x-0 h-1 bg-primary animate-scan" style={{ boxShadow: "0 0 20px var(--neon)" }} />
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="w-14 h-14 rounded-full glass flex items-center justify-center mb-3 animate-pulse-glow">
                  <Upload className="w-6 h-6 text-primary" />
                </div>
                <p className="font-medium">Glissez-déposez une radiographie</p>
                <p className="text-sm text-muted-foreground mt-1">ou cliquez pour sélectionner un fichier (PNG, JPG)</p>
              </div>
            )}
          </div>

          <div className="flex gap-3">
            <Button
              onClick={analyze}
              disabled={!file || status === "loading"}
              size="lg"
              className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 glow font-semibold"
            >
              {status === "loading" ? "Analyse en cours…" : "Analyze X-Ray"}
            </Button>
            {file && (
              <Button onClick={reset} size="lg" variant="outline" className="neon-border">
                Reset
              </Button>
            )}
          </div>

          {/* Result card */}
          {result && (
            <div
              className={`glass rounded-2xl p-6 transition-all duration-500 ${status === "pneumonia" ? "border-[oklch(0.7_0.22_25/0.6)]" : "border-[oklch(0.78_0.2_150/0.6)]"}`}
              style={{
                boxShadow:
                  status === "pneumonia"
                    ? "0 0 40px oklch(0.7 0.22 25 / 0.4)"
                    : "0 0 40px oklch(0.78 0.2 150 / 0.4)",
              }}
            >
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full flex items-center justify-center"
                     style={{ background: status === "pneumonia" ? "oklch(0.7 0.22 25 / 0.2)" : "oklch(0.78 0.2 150 / 0.2)" }}>
                  {status === "pneumonia" ? (
                    <AlertTriangle className="w-6 h-6" style={{ color: "var(--danger)" }} />
                  ) : (
                    <ShieldCheck className="w-6 h-6" style={{ color: "var(--success)" }} />
                  )}
                </div>
                <div className="flex-1">
                  <p className="text-xs uppercase tracking-widest text-muted-foreground">Résultat</p>
                  <p className="text-2xl font-bold tracking-tight"
                     style={{ color: status === "pneumonia" ? "var(--danger)" : "var(--success)" }}>
                    {result.prediction}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs uppercase tracking-widest text-muted-foreground">Confiance</p>
                  <p className="text-2xl font-bold">{(result.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
              <div className="mt-4 h-2 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full transition-all duration-700"
                  style={{
                    width: `${result.confidence * 100}%`,
                    background: status === "pneumonia" ? "var(--danger)" : "var(--success)",
                    boxShadow: `0 0 12px ${status === "pneumonia" ? "var(--danger)" : "var(--success)"}`,
                  }}
                />
              </div>
              {result.mock && (
                <p className="text-[11px] text-muted-foreground mt-3">
                  ⓘ Mode démo. Définissez <code className="text-primary">BACKEND_URL</code> pour activer l'analyse complète.
                </p>
              )}
            </div>
          )}
        </div>

        {/* RIGHT: 3D scene */}
        <div className="relative h-[500px] lg:h-[640px] glass rounded-3xl overflow-hidden glow">
          <Lung3D status={status} />
          <div className="absolute bottom-4 left-4 right-4 flex justify-between text-xs text-muted-foreground">
            <span>Live 3D Pulmonary Model</span>
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full"
                    style={{ background: status === "pneumonia" ? "var(--danger)" : status === "normal" ? "var(--success)" : "var(--neon)" }} />
              {status === "loading" ? "scanning…" : status === "pneumonia" ? "anomaly detected" : status === "normal" ? "healthy pattern" : "awaiting input"}
            </span>
          </div>
        </div>
      </section>

      {/* History */}
      {history.length > 0 && (
        <section className="container mx-auto px-6 pb-20">
          <h3 className="text-sm uppercase tracking-widest text-muted-foreground mb-4">Historique des analyses</h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {history.map((h, i) => (
              <div key={i} className="glass rounded-xl p-4 flex items-center justify-between">
                <div className="min-w-0">
                  <p className="font-medium truncate">{h.name}</p>
                  <p className="text-xs text-muted-foreground">{new Date(h.at).toLocaleTimeString()}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-bold"
                     style={{ color: h.prediction === "PNEUMONIA" ? "var(--danger)" : "var(--success)" }}>
                    {h.prediction}
                  </p>
                  <p className="text-xs text-muted-foreground">{(h.confidence * 100).toFixed(0)}%</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <footer className="container mx-auto px-6 py-8 text-center text-xs text-muted-foreground border-t border-border/40">
        PneumoScan · Outil d'aide au diagnostic — ne remplace pas un avis médical professionnel.
      </footer>
    </main>
  );
}
