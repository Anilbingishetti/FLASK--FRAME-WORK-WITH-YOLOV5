import base64
from fastapi import FastAPI, File
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from base64 import b64encode
from json import dumps, loads
from keras_segmentation.models.fcn import fcn_32_resnet50
from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.predict import predict

def get_yolov5():
    model = torch.hub.load('yolov5', 'custom', path='./static/final.pt', source='local' ,force_reload=True) 
    model.conf = 0.5
    return model

def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB").resize((640,640))
    return input_image

def get_image_from_bytes_for_segmentation(binary_image):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    print(input_image.size)
    input_image.save("tempinp.png")



model = get_yolov5()
 

app = FastAPI(
    title="Face detection api",
    description="yoo",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/notify/v1/health')
def get_health():
    return dict(msg='OK')


@app.post("/object-to-json")
async def detect_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    print(results)
    detect_res = results.pandas().xyxy[0].to_json(orient="records") 
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    # print(results)
    detect_res = results.pandas().xyxy[0].to_json(orient="records") 
    detect_res = json.loads(detect_res)
    results.render() 
    li=set() 
    for i in detect_res:
        # print(i["name"])
        li.add(i["name"])
    print(li)
    # print(type(results))
    for img in results.ims:
        pil_img=Image.fromarray(img).resize((832,480))
        pil_img.save("temp.png")
        with open('temp.png', 'rb') as open_file:
            byte_content = open_file.read()
        base64_bytes = b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')
        raw_data = {"image": base64_string}
        json_data = dumps(raw_data, indent=2)
    return {"result": li, "img":Response(content=json_data, media_type="image/jpeg")}