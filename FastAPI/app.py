from fastapi import Body
import numpy as np
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
import onnxruntime
from fastapi.responses import JSONResponse


app = FastAPI()
onnx_model_path = "random_forest_model.onnx"
sess = onnxruntime.InferenceSession(onnx_model_path)
prediction_labels = {0: 'Not Depressed', 1: 'Depressed'}
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request, "prediction_text": ""})
'''
@app.post('/predict')
async def predict(
    request: Request,
    data: dict = Body(...)
):
    Gender = data.get("Gender", 0)
    Age = data.get("Age", 0)
    Course = data.get("Course", 0)
    Year_of_Study = data.get("Year_of_Study", 0)
    CGPA = data.get("CGPA", 0.0)
    Marital_Status = data.get("Marital_Status", 0)
    Anxiety = data.get("Anxiety", 0)
    Panic_Atack = data.get("Panic_Atack", 0)
    Treatment = data.get("Treatment", 0)
    # Preprocess input data
    input_data = np.array([[Gender, Age, Course, Year_of_Study, CGPA, Marital_Status, Anxiety, Panic_Atack, Treatment]])

    # Make prediction
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_probs = sess.run([output_name], {input_name: input_data.astype(np.float32)})

    print("pred_probs shape:", pred_probs[0].shape)
    print("pred_probs values:", pred_probs[0])

    try:
        # Assuming binary classification
        prediction = int(pred_probs[0][0] > 0.5)
        predicted_label = prediction_labels[prediction]

        # Return JSON response
        return JSONResponse(content={"prediction_text": f"Prediction: {predicted_label}"})
    except IndexError:
        return JSONResponse(content={"prediction_text": "Error in prediction"})
'''
@app.post('/predict')
async def predict(
    request: Request,
    Gender: int = Form(...),
    Age: int = Form(...),
    Course: int = Form(...),
    Year_of_Study: int = Form(...),
    CGPA: float = Form(...),
    Marital_Status: int = Form(...),
    Anxiety: int = Form(...),
    Panic_Atack: int = Form(...),
    Treatment: int = Form(...),
):
    # Preprocess input data
    input_data = np.array([[Gender, Age, Course, Year_of_Study, CGPA, Marital_Status, Anxiety, Panic_Atack, Treatment]])

    # Make prediction
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_probs = sess.run([output_name], {input_name: input_data.astype(np.float32)})

    print("pred_probs shape:", pred_probs[0].shape)
    print("pred_probs values:", pred_probs[0])

    try:
        # Assuming binary classification
        prediction = int(pred_probs[0][0] > 0.5)
        predicted_label = prediction_labels[prediction]

        # Return JSON response
        return JSONResponse(content={"prediction_text": f"Prediction: {predicted_label}"})
    except IndexError:
        return JSONResponse(content={"prediction_text": "Error in prediction"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
