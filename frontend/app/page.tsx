"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Prediction = { sport: string; confidence: number };

type Analysis = {
  sport: string;
  confidence: number;
  category: string;
  origin: string;
  top_k: Prediction[];
  description: string;
  meta: {
    model: string;
    backbone: string;
    input_size: number;
    num_classes: number;
    inference_ms: number;
  };
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";
const MAX_BYTES = 5 * 1024 * 1024;

export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then((b) => setOnline(Boolean(b.model_loaded)))
      .catch(() => setOnline(false));
  }, []);

  const ingest = useCallback((chosen: File) => {
    setError(null);
    setResult(null);

    if (!chosen.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    if (chosen.size > MAX_BYTES) {
      setError("Image is larger than 5 MB. Please use a smaller file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
      setFileName(chosen.name);
      setFile(chosen);
    };
    reader.onerror = () => setError("Could not read that file.");
    reader.readAsDataURL(chosen);
  }, []);

  async function classify() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("image", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form,
      });

      const body = await response.json();
      if (!response.ok) {
        setError(body.error ?? "The request failed.");
        return;
      }
      setResult(body as Analysis);
    } catch {
      setError(
        `Could not reach the Flask backend at ${API_URL}. Start it with "python app.py".`,
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setPreview(null);
    setFileName("");
    setFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  const pct = result ? Math.round(result.confidence * 100) : 0;

  return (
    <main className="mx-auto w-full max-w-5xl px-5 py-12 sm:py-16">
      <header className="rise mb-10 text-center">
        <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium tracking-wide text-indigo-200">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              online === null
                ? "bg-slate-500"
                : online
                  ? "bg-emerald-400"
                  : "bg-red-400"
            }`}
          />
          {online === null
            ? "Connecting to model server…"
            : online
              ? "EfficientNetB0 model loaded"
              : "Model server offline"}
        </span>
        <h1 className="mt-5 text-4xl font-semibold tracking-tight text-white sm:text-5xl">
          Which sport is this?
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-slate-400 sm:text-base">
          Upload a photograph of a sport — cricket, tennis, badminton, soccer, swimming,
          karate or wrestling — and a fine-tuned EfficientNetB0 classifier will identify
          it from the image.
        </p>
      </header>

      <section className="panel rise rounded-2xl p-5 sm:p-7">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const dropped = e.dataTransfer.files?.[0];
            if (dropped) ingest(dropped);
          }}
          onClick={() => inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition ${
            dragging
              ? "border-indigo-400 bg-indigo-500/10"
              : "border-white/15 hover:border-white/30 hover:bg-white/[0.03]"
          }`}
        >
          {preview ? (
            <>
              <img
                src={preview}
                alt="Selected sport"
                className="max-h-72 w-auto rounded-lg object-contain shadow-2xl"
              />
              <p className="mt-4 text-xs text-slate-400">{fileName}</p>
            </>
          ) : (
            <>
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-white/5 text-2xl">
                ⬆
              </div>
              <p className="text-sm font-medium text-slate-200">
                Drop an image here, or click to browse
              </p>
              <p className="mt-1 text-xs text-slate-500">
                JPEG, PNG, WebP or GIF · up to 5 MB
              </p>
            </>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif"
          className="hidden"
          onChange={(e) => {
            const chosen = e.target.files?.[0];
            if (chosen) ingest(chosen);
          }}
        />

        <div className="mt-5 flex flex-col gap-3 sm:flex-row">
          <button
            onClick={classify}
            disabled={!file || loading}
            className="flex-1 rounded-xl bg-indigo-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-slate-500"
          >
            {loading ? "Running inference…" : "Identify the sport"}
          </button>
          {preview && (
            <button
              onClick={reset}
              className="rounded-xl border border-white/15 px-5 py-3 text-sm font-medium text-slate-300 transition hover:bg-white/5"
            >
              Clear
            </button>
          )}
        </div>

        {error && (
          <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </p>
        )}
      </section>

      {result && (
        <section className="rise mt-6 space-y-5">
          <div className="panel rounded-2xl p-6 sm:p-8">
            <p className="text-xs font-medium uppercase tracking-widest text-slate-500">
              Identified sport
            </p>
            <h2 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">
              {result.sport}
            </h2>

            <div className="mt-4 flex flex-wrap gap-2">
              <Tag>{result.category}</Tag>
              <Tag>{result.origin}</Tag>
            </div>

            <div className="mt-6">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span>Softmax confidence</span>
                <span className="font-semibold text-slate-200">{pct}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="bar h-full rounded-full bg-gradient-to-r from-indigo-400 to-emerald-400"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>

            <p className="mt-6 text-sm leading-relaxed text-slate-300">
              {result.description}
            </p>
          </div>

          {result.top_k?.length > 0 && (
            <div className="panel rounded-2xl p-6 sm:p-8">
              <h3 className="text-sm font-semibold text-white">
                Class probability distribution
              </h3>
              <div className="mt-4 space-y-3">
                {result.top_k.map((item, i) => (
                  <div key={i}>
                    <div className="mb-1.5 flex items-center justify-between text-sm">
                      <span className={i === 0 ? "text-white" : "text-slate-300"}>
                        {item.sport}
                      </span>
                      <span className="text-xs text-slate-500">
                        {(item.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className={`bar h-full rounded-full ${
                          i === 0
                            ? "bg-gradient-to-r from-indigo-400 to-emerald-400"
                            : "bg-slate-500"
                        }`}
                        style={{ width: `${item.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="panel rounded-2xl p-5 text-xs text-slate-500">
            <div className="flex flex-wrap gap-x-6 gap-y-2">
              <span>Model: {result.meta.model}</span>
              <span>Backbone: {result.meta.backbone}</span>
              <span>
                Input: {result.meta.input_size}×{result.meta.input_size}
              </span>
              <span>Classes: {result.meta.num_classes}</span>
              <span>Inference: {result.meta.inference_ms} ms</span>
            </div>
          </div>
        </section>
      )}

      <footer className="mt-14 text-center text-xs text-slate-600">
        Experiential Learning Group Assignment · Language Processing using ML and LLMs
      </footer>
    </main>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
      {children}
    </span>
  );
}
