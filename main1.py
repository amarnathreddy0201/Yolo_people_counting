from ultralytics import YOLO
import argparse
from tracker.bot_sort import BoTSORT

import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from fastapi import FastAPI, UploadFile, Form, Request
import numpy as np

# Initialize YOLO model
model = YOLO('yolov8s.pt')
area1 = [(100, 200), (200, 200), (200, 300), (100, 300)]  # Example coordinates for entering
area2 = [(400, 200), (500, 200), (500, 300), (400, 300)]  # Example coordinates for exiting


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    parser.add_argument("--path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    # parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    # parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    # parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    # parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser




app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")


import asyncio

# ---- Safe process_frame fix ----
def process_frame(frame, model, class_list, tracker, going_in, going_out, area1, area2, counter1, counter2):
    results = model(frame, verbose=False)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, class_ids, confs):
            # Safe index check
            if cls_id < len(model.names):
                class_name = model.names[int(cls_id)]
            else:
                continue

            if class_name not in class_list:
                continue  # skip other classes

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
    return frame


# # ---- Async + Thread-safe frame generator ----
# async def generate_frames(source):
#     yield from await asyncio.to_thread(generate_frames_sync, source)

# ---- Async + Thread-safe frame generator ----
async def generate_frames(source):
    loop = asyncio.get_event_loop()
    # Use run_in_executor to call the blocking generator in another thread
    for frame in await loop.run_in_executor(None, lambda: list(generate_frames_sync(source))):
        yield frame


def generate_frames_sync(source):
    args = make_parser().parse_args([])
    tracker = BoTSORT(args)
    cap = cv2.VideoCapture(source)
    class_list = ['person']
    counter1, counter2, going_in, going_out = [], [], [], []

    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # frame = process_frame(
        #     frame,
        #     model=model,
        #     class_list=class_list,
        #     tracker=tracker,
        #     going_in=going_in,
        #     going_out=going_out,
        #     area1=area1,
        #     area2=area2,
        #     counter1=counter1,
        #     counter2=counter2,
        # )

        # Resize if needed
        frame = cv2.resize(frame, (640, 480))


        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

# ----------- HTML Page -----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# # ----------- Webcam stream -----------
# @app.get("/webcam_feed")
# def webcam_feed():
#     return StreamingResponse(generate_frames(0),
#                              media_type="multipart/x-mixed-replace; boundary=frame")

# ----------- Upload video -----------
@app.post("/upload_video")
async def upload_video(file: UploadFile):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return JSONResponse({"filename": file.filename})

@app.get("/webcam_feed")
async def webcam_feed():
    return StreamingResponse(
        generate_frames(0),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/play_video")
async def play_video(filename: str):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return StreamingResponse(
        generate_frames(path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/rtsp_stream")
async def rtsp_stream(rtsp_url: str = Form(...)):
    return StreamingResponse(
        generate_frames(rtsp_url),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
