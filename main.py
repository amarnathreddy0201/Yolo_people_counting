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

    parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
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
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser




app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color
import numpy as np

def process_frame(frame, model, class_list, tracker, area1, area2, going_out, going_in, counter1, counter2,args):
    # YOLOv8 detection
    # results = model.predict(frame)
    # boxes_data = results[0].boxes.data
    
    # detected_objects = []
    # for row in boxes_data:
    #     # Extract bounding box and class information
    #     x1, y1, x2, y2, _, d = map(int, row)
    #     if 'person' in class_list[d]:
    #         detected_objects.append([x1, y1, x2 - x1, y2 - y1])  # Add bounding box

    results = model(frame, classes=0)
    detected_objects=[]
    if results and len(results[0].boxes) > 0:
                
        for d in results[0].boxes:
            
            clas = int(d.cls.item())  # Convert to integer
            
            conf = d.conf.item()  # Convert to float
            x1, y1, x2, y2 = map(int, d.xyxy[0])

            # tracker_input.append(np.ndarray([x1,y1,x2,y2,conf,clas]))
            detected_objects.append(np.array([x1, y1, x2, y2, conf, clas]))
    
    # online_targets = tracker.update(detected_objects,frame)
    online_targets = tracker.update(np.array(detected_objects,dtype=np.float64),frame)


    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
        # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
        online_tlwhs.append(tlwh)
        online_ids.append(tid)
        online_scores.append(t.score)
    # Check if people cross the defined areas

    # for bbox in online_tlwhs:
    #     # tlwh 
    #     # obj_id = bbox

    #     x3, y3, x4, y4 = bbox
    #     intbox = tuple(map(int, (x1, y1, x1 + x4, y1 + y4)))
    #     obj_id = int(online_ids[i])


    #     # x3, y3, x4, y4 = bbox.tlwh

    #     if cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
    #         going_out[obj_id] = (x4, y4)
    #     if obj_id in going_out and cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
    #         if obj_id not in counter1:
    #             counter1.append(obj_id)

    #     if cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
    #         going_in[obj_id] = (x4, y4)
    #     if obj_id in going_in and cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
    #         if obj_id not in counter2:
    #             counter2.append(obj_id)

    # cv2.putText(frame, f'In: {counter2}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame, f'Out: {counter1}', (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                


    # Count logic
    for i, tlwh in enumerate(online_tlwhs):
        x, y, w, h = tlwh
        cx, cy = int(x + w / 2), int(y + h / 2)
        obj_id = online_ids[i]

        # Draw ID and bounding box
        color = get_color(obj_id)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(frame, f"ID {obj_id}", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Check if inside area1 or area2
        in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0
        in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0

        # Track movements
        if in_area1:
            going_in[obj_id] = (cx, cy)
        elif in_area2:
            going_out[obj_id] = (cx, cy)

        # Check transitions
        # OUT: area2 → area1
        if obj_id in going_out and in_area1:
            if obj_id not in counter1:
                counter1.append(obj_id)

        # IN: area1 → area2
        if obj_id in going_in and in_area2:
            if obj_id not in counter2:
                counter2.append(obj_id)

    # Draw areas
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

    # Display counters
    cv2.putText(frame, f'IN: {len(counter2)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f'OUT: {len(counter1)}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return frame
    # return frame #, len(counter1), len(counter2)



# ----------- Stream generator -----------
def generate_frames(source):
    args = make_parser().parse_args()
    tracker = BoTSORT(args)
    cap = cv2.VideoCapture(source)
    class_list = ['person']
    counter1=[]
    counter2=[]

    going_in=[]
    going_out=[]

    model = YOLO("yolov8s.pt")
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = process_frame(frame,args=args,model=model,class_list=class_list,tracker = tracker,going_in=going_in,going_out=going_out,area1=area1,area2=area2,counter1=counter1,counter2=counter2)





        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    cap.release()

# ----------- HTML Page -----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ----------- Webcam stream -----------
@app.get("/webcam_feed")
def webcam_feed():
    return StreamingResponse(generate_frames(0),
                             media_type="multipart/x-mixed-replace; boundary=frame")

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
    return StreamingResponse(generate_frames(path),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ----------- RTSP stream -----------
@app.post("/rtsp_stream")
async def rtsp_stream(rtsp_url: str = Form(...)):
    return StreamingResponse(generate_frames(rtsp_url),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
