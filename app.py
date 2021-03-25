from urllib import request
from fastapi import FastAPI, Request
from fastai.vision.all import load_learner, PILImage, Path
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from urllib.request import urlretrieve
from os import remove

app = FastAPI()


templates = Jinja2Templates(directory='.')

def predict(filename: str):
    '''classifying image.'''
    model = load_learner(Path.cwd()/'bear.pkl')
    img = PILImage.create(filename)
    pred_class, pred_idx, ful_tensor = model.predict(img)
    return str(pred_class)


@app.get("/")
async def home(request: Request):
    '''home page'''
    return templates.TemplateResponse('index.html', context={'request': request, 'cap': 'upload picture'})


@app.post("/uploadfile")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    '''Uploading file'''
    print(file.filename)
    
    if 'image' in file.content_type:
        contents = await file.read()

        with open(file.filename, 'wb') as f:
            f.write(contents)
        
        prediction = predict(file.filename)
        print(prediction)
        remove(file.filename)

    return templates.TemplateResponse('index.html', context={'request':request, 'cap': f'predicted result is: {prediction}', 'another': "Click below button to upload new."})
