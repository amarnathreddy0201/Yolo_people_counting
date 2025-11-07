# import cv2
# import numpy as np
# from ultralytics import YOLO
# from tracker.bot_sort import BoTSORT
# import argparse

# # Initialize YOLO
# model = YOLO("yolov8s.pt")

# # Line coordinates (y = 400)
# LINE_POSITION = 400

# # BoT-SORT parser
# def make_parser():
#     parser = argparse.ArgumentParser("BoT-SORT tracking")
    
#     parser.add_argument("--path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
#     parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
#     parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
#     parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
#     # parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
#     parser.add_argument("-expn", "--experiment-name", type=str, default=None)
#     parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
#     parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

#     # Detector
#     parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
#     parser.add_argument("--conf", default=None, type=float, help="test conf")
#     parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
#     parser.add_argument("--tsize", default=None, type=int, help="test img size")
#     parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
#     parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

#     # tracking args
#     parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
#     parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
#     parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
#     parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

#     # CMC
#     parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

#     # ReID
#     parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
#     parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
#     parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
#     parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
#     parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

#     return parser

# # Color function
# def get_color(idx):
#     idx *= 3
#     return ((37*idx)%255, (17*idx)%255, (29*idx)%255)

# # Process frame
# def process_frame(frame, tracker, counter_in, counter_out, args):
#     results = model(frame, classes=0)
#     detections = []

#     if results and len(results[0].boxes) > 0:
#         for d in results[0].boxes:
#             conf = d.conf.item()
#             x1, y1, x2, y2 = map(int, d.xyxy[0])
#             detections.append([x1, y1, x2, y2, conf, 0])

#     online_targets = tracker.update(np.array(detections, dtype=np.float64), frame)

#     if not hasattr(process_frame, "prev_positions"):
#         process_frame.prev_positions = {}

#     for t in online_targets:
#         tlwh = t.tlwh
#         tid = t.track_id
#         x, y, w, h = tlwh
#         cx, cy = int(x + w/2), int(y + h/2)

#         # Draw box and ID
#         color = get_color(tid)
#         cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
#         cv2.putText(frame, f"ID {tid}", (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Previous position
#         prev_y = process_frame.prev_positions.get(tid, cy)
#         process_frame.prev_positions[tid] = cy

#         # Count logic based on line crossing
#         if tid not in counter_in and prev_y < LINE_POSITION <= cy:
#             counter_in.add(tid)  # top → bottom
#         elif tid not in counter_out and prev_y > LINE_POSITION >= cy:
#             counter_out.add(tid)  # bottom → top

#     # Draw line
#     cv2.line(frame, (LINE_POSITION,0), (LINE_POSITION,frame.shape[0]), (0, 0, 255), 2)

#     # Display counts
#     cv2.putText(frame, f'IN: {len(counter_in)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
#     cv2.putText(frame, f'OUT: {len(counter_out)}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)

#     return frame

# # Main video loop
# def main(video_path):
#     args = make_parser().parse_args([])
#     tracker = BoTSORT(args)
#     cap = cv2.VideoCapture(video_path)

#     counter_in = set()
#     counter_out = set()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = process_frame(frame, tracker, counter_in, counter_out, args)
#         cv2.imshow("People Counting", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"✅ Total IN: {len(counter_in)}, OUT: {len(counter_out)}")

# if __name__ == "__main__":
#     main("/home/mantra/development/Yolo_people_counting/p.mp4")


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


# Initialize YOLO
model = YOLO("yolov8s.pt")

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Vertical line x-coordinate
LINE_POSITION = 500  # Change according to your frame

# BoT-SORT parser (keep all original argument names)
def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    parser.add_argument("--path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
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
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser

# Color function
def get_color(idx):
    idx *= 3
    return ((37*idx)%255, (17*idx)%255, (29*idx)%255)

# Process frame with vertical line logic
def process_frame(frame, tracker, counter_in, counter_out, args):
    results = model(frame, classes=0)
    detections = []

    if results and len(results[0].boxes) > 0:
        for d in results[0].boxes:
            conf = d.conf.item()
            x1, y1, x2, y2 = map(int, d.xyxy[0])
            detections.append([x1, y1, x2, y2, conf, 0])

    online_targets = tracker.update(np.array(detections, dtype=np.float64), frame)

    # Initialize previous positions and flags
    if not hasattr(process_frame, "prev_positions"):
        process_frame.prev_positions = {}
    if not hasattr(process_frame, "object_flags"):
        process_frame.object_flags = {}

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        x, y, w, h = tlwh
        cx, cy = int(x + w/2), int(y + h/2)

        # Draw bounding box and ID
        color = get_color(tid)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(frame, f"ID {tid}", (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Previous position
        prev_x = process_frame.prev_positions.get(tid, cx)
        process_frame.prev_positions[tid] = cx

        # Initialize flags if not exists
        if tid not in process_frame.object_flags:
            process_frame.object_flags[tid] = {"counted_in": False, "counted_out": False}

        # Count logic
        # Left → Right = IN
        if prev_x < LINE_POSITION <= cx and not process_frame.object_flags[tid]["counted_in"]:
            counter_in.add(tid)
            process_frame.object_flags[tid]["counted_in"] = True
        # Right → Left = OUT
        elif prev_x > LINE_POSITION >= cx and not process_frame.object_flags[tid]["counted_out"]:
            counter_out.add(tid)
            process_frame.object_flags[tid]["counted_out"] = True

    # Draw vertical line
    cv2.line(frame, (LINE_POSITION,0), (LINE_POSITION,frame.shape[0]), (0,0,255), 2)

    # Display counts
    cv2.putText(frame, f'IN: {len(counter_in)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
    cv2.putText(frame, f'OUT: {len(counter_out)}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)

    return frame

# Main video loop
def main(video_path):
    args = make_parser().parse_args([])
    tracker = BoTSORT(args)
    cap = cv2.VideoCapture(video_path)

    counter_in = set()
    counter_out = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, tracker, counter_in, counter_out, args)
        # cv2.imshow("People Counting", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Total IN (Left→Right): {len(counter_in)}, OUT (Right→Left): {len(counter_out)}")

# if __name__ == "__main__":
#     main("p.mp4")


# ----------- HTML Page -----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ----------- Webcam stream -----------
# @app.get("/webcam_feed")
# def webcam_feed():
#     return StreamingResponse(main(0),
#                              media_type="multipart/x-mixed-replace; boundary=frame")

# ----------- Upload video -----------
@app.post("/upload_video")
async def upload_video(file: UploadFile):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return JSONResponse({"filename": file.filename})

@app.get("/play_video")
def play_video(filename: str):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return StreamingResponse(main(path),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ----------- RTSP stream -----------
# @app.post("/rtsp_stream")
# async def rtsp_stream(rtsp_url: str = Form(...)):
#     return StreamingResponse(main(rtsp_url),
#                              media_type="multipart/x-mixed-replace; boundary=frame")

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
