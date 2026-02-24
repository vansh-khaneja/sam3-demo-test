"use client";

import { useState, useRef, useEffect } from "react";

const API_URL = "https://px9zo05g6n5x26-3000.proxy.runpod.net";

type Bbox = { x1: number; y1: number; x2: number; y2: number };
type Result = { prompt: string; count: number; colorIdx: number };

const COLORS = [
  { hex: "#06b6d4", r: 6, g: 182, b: 212 },   // cyan-500
  { hex: "#8b5cf6", r: 139, g: 92, b: 246 },   // violet-500
  { hex: "#22c55e", r: 34, g: 197, b: 94 },      // green-500
  { hex: "#f43f5e", r: 244, g: 63, b: 94 },     // rose-500
];

function drawOnCanvas(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  b: Bbox | null,
  colorIdx: number = 0
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const c = COLORS[colorIdx % COLORS.length];

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (!b) return;

  // Light dim outside
  ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},0.15)`;
  ctx.fillRect(0, 0, canvas.width, b.y1);
  ctx.fillRect(0, b.y2, canvas.width, canvas.height - b.y2);
  ctx.fillRect(0, b.y1, b.x1, b.y2 - b.y1);
  ctx.fillRect(b.x2, b.y1, canvas.width - b.x2, b.y2 - b.y1);

  // Border
  ctx.strokeStyle = c.hex;
  ctx.lineWidth = 2;
  ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [conf, setConf] = useState(0.65);
  const [results, setResults] = useState<Result[]>([]);
  const [colorIdx, setColorIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const originalImgRef = useRef<HTMLImageElement | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const scaleRef = useRef(1);
  const drawingRef = useRef(false);
  const startRef = useRef({ x: 0, y: 0 });
  const bboxRef = useRef<Bbox | null>(null);
  const pulseRef = useRef(0);
  const resultBlobRef = useRef<Blob | null>(null);

  useEffect(() => {
    if (!preview) return;
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      originalImgRef.current = img;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const container = canvas.parentElement;
      const maxW = (container?.clientWidth || 900) - 16;
      const scale = Math.min(maxW / img.width, 600 / img.height, 1);
      scaleRef.current = scale;
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      drawOnCanvas(canvas, img, null);
    };
    img.src = preview;
  }, [preview]);

  const getPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (loading) return;
    startRef.current = getPos(e);
    drawingRef.current = true;
    bboxRef.current = null;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const pos = getPos(e);
    const s = startRef.current;
    const b: Bbox = {
      x1: Math.min(s.x, pos.x),
      y1: Math.min(s.y, pos.y),
      x2: Math.max(s.x, pos.x),
      y2: Math.max(s.y, pos.y),
    };
    bboxRef.current = b;
    drawOnCanvas(canvas, img, b, colorIdx);
  };

  const handleMouseUp = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    const b = bboxRef.current;
    if (b && b.x2 - b.x1 > 10 && b.y2 - b.y1 > 10) {
      runSegmentation(b);
    }
  };

  const startPulse = (b: Bbox) => {
    let angle = 0;
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const c = COLORS[colorIdx % COLORS.length];

    const w = b.x2 - b.x1;
    const h = b.y2 - b.y1;
    const perimeter = 2 * (w + h);

    const getPoint = (dist: number): [number, number] => {
      const d = ((dist % perimeter) + perimeter) % perimeter;
      if (d < w) return [b.x1 + d, b.y1];
      if (d < w + h) return [b.x2, b.y1 + (d - w)];
      if (d < 2 * w + h) return [b.x2 - (d - w - h), b.y2];
      return [b.x1, b.y2 - (d - 2 * w - h)];
    };

    const animate = () => {
      drawOnCanvas(canvas, img, b, colorIdx);

      const headDist = angle;
      const tailLen = perimeter * 0.35;
      const steps = 50;

      for (let i = 0; i < steps; i++) {
        const t = i / steps;
        const d = headDist - t * tailLen;
        const [x1, y1] = getPoint(d);
        const [x2, y2] = getPoint(d - tailLen / steps);
        const alpha = (1 - t) * 0.9;
        const r = Math.round(c.r + t * 20);
        const g = Math.round(c.g + t * 30);
        const b2 = Math.round(c.b + t * 20);
        ctx.strokeStyle = `rgba(${r},${g},${b2},${alpha})`;
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      angle = (angle + 2) % perimeter;
      pulseRef.current = requestAnimationFrame(animate);
    };
    pulseRef.current = requestAnimationFrame(animate);
  };

  const stopPulse = () => cancelAnimationFrame(pulseRef.current);

  const runSegmentation = async (b: Bbox) => {
    if (!image) return;
    setLoading(true);
    startPulse(b);

    const scale = scaleRef.current;
    const fd = new FormData();
    fd.append("image", image);
    fd.append("x1", Math.round(b.x1 / scale).toString());
    fd.append("y1", Math.round(b.y1 / scale).toString());
    fd.append("x2", Math.round(b.x2 / scale).toString());
    fd.append("y2", Math.round(b.y2 / scale).toString());
    fd.append("conf", conf.toString());
    fd.append("color_index", colorIdx.toString());

    // Send previous result as base so masks accumulate
    if (resultBlobRef.current) {
      fd.append("base_image", resultBlobRef.current, "base.png");
    }

    console.log("Bounding box selected:", b);
    console.log("File upload complete. FormData keys:", Array.from(fd.keys()));

    try {
      const startTime = Date.now();
      const res = await fetch(`${API_URL}/auto-segment`, {
        method: "POST",
        body: fd,
      });
      const endTime = Date.now();
      console.log("API /auto-segment time:", endTime - startTime, "ms");
      stopPulse();
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");

      if (!res.ok) {
        if (canvas && imgRef.current)
          drawOnCanvas(canvas, imgRef.current, b, colorIdx);
        setLoading(false);
        return;
      }

      const count = Number(res.headers.get("X-Contours") || 0);
      const prompt = res.headers.get("X-Prompt") || "";

      setResults((prev) => [...prev, { prompt, count, colorIdx }]);

      const blob = await res.blob();
      resultBlobRef.current = blob;

      const url = URL.createObjectURL(blob);
      const ri = new Image();
      ri.onload = () => {
        if (!canvas || !ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(ri, 0, 0, canvas.width, canvas.height);
        imgRef.current = ri; // use result as new base for next bbox
        setShowResult(true);
        setColorIdx((prev) => (prev + 1) % COLORS.length);
      };
      ri.src = url;
    } catch {
      stopPulse();
      const canvas = canvasRef.current;
      if (canvas && imgRef.current)
        drawOnCanvas(canvas, imgRef.current, b, colorIdx);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setShowResult(false);
    setResults([]);
    setColorIdx(0);
    bboxRef.current = null;
    resultBlobRef.current = null;
    imgRef.current = originalImgRef.current;
    const canvas = canvasRef.current;
    if (canvas && originalImgRef.current)
      drawOnCanvas(canvas, originalImgRef.current, null);
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    bboxRef.current = null;
    resultBlobRef.current = null;
    setResults([]);
    setColorIdx(0);
    setShowResult(false);
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-10 font-[family-name:var(--font-geist-sans)]">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-lg font-semibold text-slate-800">SAM3 Segment</h1>
      </div>

      {/* Controls */}
      <div className="mb-3 flex items-center gap-3">
        <button
          onClick={() => fileRef.current?.click()}
          className="rounded-md bg-blue-500 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-blue-600"
        >
          {image ? "Change image" : "Upload image"}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="hidden"
        />

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Confidence</span>
          <input
            type="range"
            min={0.1}
            max={0.99}
            step={0.01}
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="font-[family-name:var(--font-geist-mono)] text-xs text-slate-500">
            {conf.toFixed(2)}
          </span>
        </div>

        <div className="flex-1" />

        {results.length > 0 && (
          <button
            onClick={handleReset}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-500 transition hover:bg-white hover:text-slate-700"
          >
            Reset
          </button>
        )}
      </div>

      {/* Canvas */}
      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        {!preview ? (
          <div
            onClick={() => fileRef.current?.click()}
            className="flex h-[420px] cursor-pointer flex-col items-center justify-center gap-2 transition hover:bg-blue-50/50"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50">
              <svg
                className="h-5 w-5 text-blue-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 4.5v15m7.5-7.5h-15"
                />
              </svg>
            </div>
            <p className="text-sm text-slate-400">Upload an image</p>
            <p className="text-xs text-slate-300">
              Then draw a box to segment
            </p>
          </div>
        ) : (
          <div className="relative flex items-center justify-center p-2">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              className={`block rounded-lg ${
                loading ? "cursor-wait" : "cursor-crosshair"
              }`}
            />
            {results.length > 0 && !loading && (
              <>
                <div className="absolute top-4 left-4 flex flex-col gap-2">
                  {results.map((r, i) => (
                    <span
                      key={i}
                      style={{
                        backgroundColor:
                          COLORS[r.colorIdx % COLORS.length].hex,
                      }}
                      className="rounded-full px-4 py-1.5 text-sm font-bold uppercase tracking-wide text-white shadow-md"
                    >
                      {r.prompt}
                    </span>
                  ))}
                </div>
                <div className="absolute top-4 right-4 flex flex-col gap-2">
                  {results.map((r, i) => (
                    <span
                      key={i}
                      style={{
                        backgroundColor:
                          COLORS[r.colorIdx % COLORS.length].hex,
                      }}
                      className="rounded-full px-4 py-1.5 text-sm font-bold uppercase tracking-wide text-white shadow-md"
                    >
                      {r.count} Items
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Status */}
      {preview && !showResult && !loading && (
        <p className="mt-2 text-center text-xs text-slate-400">
          Draw a box around the object
        </p>
      )}
      {loading && (
        <p className="mt-2 text-center text-xs text-blue-400">Processing...</p>
      )}
    </div>
  );
}
