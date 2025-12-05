import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from tensorflow import keras

app = FastAPI()
templates = Jinja2Templates(directory="templates")
MODEL_PATH = "iris_model.keras"
class_names = ['Setosa (Щетинистый)', 'Versicolor (Разноцветный)', 'Virginica (Виргинский)']

try:
    model = keras.models.load_model(MODEL_PATH)
    print("Keras модель успешно загружена!")
except Exception as e:
    model = None
    print(f"Ошибка загрузки модели: {e}. Сначала запустите train_model.py")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    if model is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": "Ошибка: Модель не найдена!"
        })

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    predictions = model.predict(input_data)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    result_text = class_names[predicted_class_index]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result_text
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
