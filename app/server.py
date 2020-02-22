from __future__ import print_function
import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from glob import glob
import sys
import webbrowser
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 

export_detection_url = 'https://drive.google.com/u/0/uc?id=1uG4-I8iGBprnBVQEMVb8mQek1TxHr6W8'
export_detection_name = 'chessboard_detection.pkl'

export_recognition_url = 'https://drive.google.com/u/0/uc?id=14UX6--3Y2ufkRvWAhPY_4VN3d1eqG_8c'
export_recognition_name = 'chessboard_recognition.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner(export_file_url , export_file_name):
    await download_file(export_file_url, path / export_file_name) 
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner(export_detection_url,export_detection_name))]
learn_detection = loop.run_until_complete(asyncio.gather(*tasks))[0]
tasks = [asyncio.ensure_future(setup_learner(export_recognition_url,export_recognition_name))]
learn_recognition = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = BytesIO(img_bytes)
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
    board_orientation = img_data['board_orientation']
    movesNext = img_data['movesNext']
    fen = getPosition(img,movesNext,board_orientation)
    return JSONResponse({'result': str(fen)})

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(image):
    img = cv2.GaussianBlur(image, (5, 5), 0)
    
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                (x, y, w, h) = cv2.boundingRect(cnt)
                ar = w / float(h)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and ar >= 0.95 and ar <= 1.05:
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    squares_arr = []
    for c in squares:
        x,y,w,h = cv2.boundingRect(c)
        squares_arr.append(image[y:y+h, x:x+h])
    return squares_arr;



def ChessBoardRecognition(image):
    candidates=[]
    i=0
    for square in find_squares(image):
        i+=1
        square_ai = Image(pil2tensor(square, dtype=np.float32).div_(255))
        pred_class, pred_idx, outputs = learn_detection.predict(square_ai)
        if pred_class.obj == 'chess position':
            candidates.append(square)
    if(len(candidates)==0):
        raise ValueError('No Board found')
    candidates.sort(key=lambda x: x.shape, reverse=True)
    return candidates[0]

def SplitBoardIntoSquares(image):
    img = ChessBoardRecognition(image)
    #loadImage(img)
    pieces_images=[]
    img = format_img(img)
    row = img.shape[0]
    col = img.shape[1]

    for r in range(0,row,int(row/8)):
        for c in range(0,col,int(col/8)):
            pieces_images.append(img[r:r+int(row/8), c:c+int(col/8),:])
    return pieces_images;
def ChessPiecesRecognition(image):   
    fen_arr=[]
    for img in SplitBoardIntoSquares(image):
        img_ai = Image(pil2tensor(img, dtype=np.float32).div_(255))
        pred_class, pred_idx, outputs = learn_recognition.predict(img_ai)
        fen_arr.append(pred_class.obj)
    return fen_arr
#Translate into FEN
classes = {'B':'White_Bishop', 'K':'White_King', 'N':'White_Knight', 'P':'White_Pawn', 'Q':'White_Queen', 'R':'White_Rook',
           'b':'Black_Bishop', 'k':'Black_King', 'n':'Black_Knight', 'p':'Black_Pawn', 'q':'Black_Queen', 'r':'Black_Rook','none':'Empty'}
def get_key(val): 
    for key, value in classes.items(): 
         if val == value: 
            return key 
  
    return "key doesn't exist"
def get_fen(image):
    
    fen=''
    c=0
    count=0
    for i in ChessPiecesRecognition(image):
        if count==8:
            if(c>0):
                fen+=str(c)
                c=0
            fen+='/'
            count=0
        if(i=='Empty'):
            c+=1
            if c==8:
                fen+=str(c)
                c=0

        else:
            if(c>0):
                fen+=str(c)
                c=0
            fen+=get_key(i)
        count+=1
    if(c>0):
        fen+=str(c)
    return fen
    

def format_img(img):
    img = cv2.resize(img,(400,400))
    return img

def getPosition(imagePath , movesNext,board_orientation='auto_detect'):
    fen = get_fen(imagePath)
    if(board_orientation=='auto_detect'):
        white=0
        line=0
        for i in fen :
            if i.isupper():
                white+=1
            if i=='/':
                line+=1
            if line==4:
                line=-1
                semi = white
                white=0
    else:
        semi=white=0
    if(semi>white or board_orientation=='black'):    
        fen = fen[::-1]
    fen += " "+movesNext[0]
    webbrowser.open('https://lichess.org/editor/'+fen)
    return fen


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
