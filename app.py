import base64
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from openai import OpenAI
from ultralytics.models.sam.predict import SAM3SemanticPredictor

load_dotenv()

app = FastAPI(title="SAM3 Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Segments", "X-Contours", "X-Prompt"],
)

# Load predictor once at startup
overrides = dict(
    conf=0.65,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)
openai_client = OpenAI()

# Color palette: BGR values matched to frontend Tailwind colors
PALETTE_BGR = [
    [212, 182, 6],    # cyan-500   (#06b6d4)
    [246, 92, 139],   # violet-500 (#8b5cf6)
    [74, 222, 34],    # green-500  (#22c55e)
    [94, 63, 244],    # rose-500   (#f43f5e)
]


def describe_crop(img_bgr: np.ndarray) -> str:
    """Send cropped image to OpenAI vision and get a short description."""
    _, buffer = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buffer).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the main object in this image in 3-5 words including its shape and relative size. For example: 'large round red apple', 'small rectangular black phone', 'tall cylindrical metal bottle'.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()


def make_overlay(
    base_img: np.ndarray, masks: np.ndarray, color_index: int = 0
) -> tuple[np.ndarray, int, int]:
    """Create mask overlay image, return (image, num_masks, num_contours)."""
    color = PALETTE_BGR[color_index % len(PALETTE_BGR)]
    num_masks = masks.shape[0]
    combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
    combined_mask = cv2.resize(
        combined_mask, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    mask_img = base_img.copy()
    overlay = mask_img.copy()
    overlay[combined_mask == 1] = (
        overlay[combined_mask == 1] * 0.5 + np.array(color) * 0.5
    ).astype(np.uint8)
    mask_img = overlay

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_img, contours, -1, color, thickness=4)

    return mask_img, num_masks, len(contours)


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    conf: float = Form(0.65),
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    predictor.args.conf = conf
    predictor.set_image(img)
    results = predictor(text=[prompt])

    if not results or results[0].masks is None:
        return {"error": "No masks detected", "prompt": prompt}

    mask_img, num_masks, num_contours = make_overlay(
        results[0].orig_img, results[0].masks.data.cpu().numpy()
    )

    _, buffer = cv2.imencode(".png", mask_img)
    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        headers={
            "X-Segments": str(num_masks),
            "X-Contours": str(num_contours),
            "X-Prompt": prompt,
        },
    )


@app.post("/auto-segment")
async def auto_segment(
    image: UploadFile = File(...),
    base_image: Optional[UploadFile] = File(None),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    conf: float = Form(0.65),
    color_index: int = Form(0),
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Use base_image for overlay if provided (accumulate mode)
    if base_image:
        base_contents = await base_image.read()
        base_nparr = np.frombuffer(base_contents, np.uint8)
        overlay_base = cv2.imdecode(base_nparr, cv2.IMREAD_COLOR)
    else:
        overlay_base = img

    # Crop bbox region from ORIGINAL image and get description
    cropped = img[y1:y2, x1:x2]
    description = describe_crop(cropped)

    # Run SAM3 on ORIGINAL image
    predictor.args.conf = conf
    predictor.set_image(img)
    results = predictor(text=[description])

    if not results or results[0].masks is None:
        return {"error": "No masks detected", "prompt": description}

    # Overlay masks on base_image (which may have previous results)
    mask_img, num_masks, num_contours = make_overlay(
        overlay_base, results[0].masks.data.cpu().numpy(), color_index
    )

    _, buffer = cv2.imencode(".png", mask_img)
    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        headers={
            "X-Segments": str(num_masks),
            "X-Contours": str(num_contours),
            "X-Prompt": description,
        },
    )
