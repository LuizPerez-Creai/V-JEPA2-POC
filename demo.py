import cv2
import torch
from transformers import AutoVideoProcessor, AutoModelForVideoClassification
from PIL import Image
import threading
import os

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model repository (use your fine-tuned folder here if you have one, e.g. "./vjepa2-ucf101-ft")
REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
print(f"Loading model '{REPO}' on device {DEVICE}...")

# Load model and processor
model = AutoModelForVideoClassification.from_pretrained(REPO).to(DEVICE).eval()
processor = AutoVideoProcessor.from_pretrained(REPO)

# ---- Load labels from the model and persist them --------------------------------

id2label = model.config.id2label or {}

label_list = [id2label[i] for i in sorted(id2label.keys(), key=int)] if id2label else []

# Save labels to a text file (one per line)
with open("model_labels.txt", "w", encoding="utf-8") as f:
    for name in label_list:
        f.write(name + "\n")

print(f"Loaded {len(label_list)} labels. First 10: {label_list[:10] if label_list else 'N/A'}")
print("All labels saved to model_labels.txt")

# Parameters
FPC = getattr(model.config, "num_frames", 16)  
frame_skip = 12       
frame_count = 0      
predictions = []      
lock = threading.Lock()  
show_labels_panel = False 

# Webcam setup
cap = cv2.VideoCapture(0)
frames = []

def inference_worker(frames_clip):
    # Use 'images=' for a list of PIL frames
    inputs = processor(videos=frames_clip, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        # AMP on CUDA for a small speed bump
        autocast_ok = (DEVICE == "cuda")
        with torch.cuda.amp.autocast(enabled=autocast_ok):
            outputs = model(**inputs)

    probs = outputs.logits.softmax(dim=-1)[0]
    topk = torch.topk(probs, k=min(3, probs.shape[-1]))
    labels = [model.config.id2label[int(i)] for i in topk.indices.tolist()]
    scores = [f"{s:.2f}" for s in topk.values.tolist()]

    with lock:
        predictions.clear()
        predictions.extend(zip(labels, scores))

# Inference thread handle
inference_thread = None

def draw_labels_panel(frame, labels_to_show, max_rows=20, x=10, y=30, w=380, line_h=22):
    """Draw a simple semi-transparent panel listing labels (alphabetical)."""
    overlay = frame.copy()
    h = min(max_rows, len(labels_to_show)) * line_h + 20
    # Panel background
    cv2.rectangle(overlay, (x - 5, y - 20), (x + w, y - 20 + h), (0, 0, 0), -1)
    # Blend
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Title
    cv2.putText(frame, f"Labels (showing up to {max_rows}/{len(labels_to_show)})", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y_text = y + 10

    for i, name in enumerate(labels_to_show[:max_rows], start=1):
        cv2.putText(frame, f"{i:>2}. {name}", (x, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        y_text += line_h

print("Press 'L' to toggle the labels panel. Press 'ESC' to exit.")

# Main loop
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1

    # Resize for inference path (processor will also handle resize/center crop)
    # Keeping it small reduces copying cost a bit for the PIL conversion.
    resized = cv2.resize(frame, (256, 256))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb)
    frames.append(pil_frame)

    # Keep only last FPC frames
    if len(frames) > FPC:
        frames = frames[-FPC:]

    # Launch inference thread on schedule
    if frame_count % frame_skip == 0 and len(frames) == FPC:
        if inference_thread is None or not inference_thread.is_alive():
            inference_thread = threading.Thread(target=inference_worker, args=(frames.copy(),), daemon=True)
            inference_thread.start()

    # Draw predictions (if any) on the *original* (non-resized) display frame
    with lock:
        y_draw = 30
        for label, score in predictions:
            cv2.putText(frame, f"{label}: {score}", (10, y_draw),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            y_draw += 30

    # Optional: draw a labels panel (alphabetical)
    if show_labels_panel and label_list:
        labels_alpha = sorted(label_list)
        draw_labels_panel(frame, labels_alpha, max_rows=20)

    # Show frame
    cv2.imshow("V-JEPA 2 — Async Inference (ESC to exit, L = labels)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key in (ord('l'), ord('L')):
        show_labels_panel = not show_labels_panel

# Cleanup
cap.release()
cv2.destroyAllWindows()
