import glob
import os

from src.models.predict_model import FitnessTrackerPredictor
from flask import Flask, render_template, request


app = Flask(
    __name__,
    static_folder="D:/Programing/Projects/Fitness-tracker-based-on-ML-2/static",
)
app.config["DEBUG"] = True


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["GET", "POST"])
def predict():

    
    

    output_dir = 'static\pred'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob("static\pred\*")


    files = glob.glob("./static/pred/*")

    for f in files:
        os.remove(f)

    accfile = request.files["accfile"]

    acc_path = "static\pred" + accfile.filename
    accfile.save(acc_path)

    gyrfile = request.files["gyrfile"]
    gyr_path = "static\pred\\" + gyrfile.filename

    acc_path = "./static/pred/" + accfile.filename
    accfile.save(acc_path)

    gyrfile = request.files["gyrfile"]
    gyr_path = "./static/pred/" + gyrfile.filename

    gyrfile.save(gyr_path)

    # Model paths
    model_path = "models/final_model.pkl"
    cluster_model_path = "models/Clustering_model.pkl"

    tracker_predictor = FitnessTrackerPredictor(
        acc_path, gyr_path, model_path, cluster_model_path
    )

    # Predict activity
    label = tracker_predictor.predict_activity()

    if label == "bench":
        prediction = "Bench Press"

        img = "\static\web\Barbell Bench Press.gif"

        img = "./static/web/Barbell Bench Press.gif"

    elif label == "squat":
        prediction = "Squat"
        img = "./static/web/BARBELL SQUAT.gif"
    elif label == "row":
        prediction = "Bent-Over Row"
        img = "./static/web/Barbell-Bent-Over-Row.gif"
    elif label == "ohp":
        prediction = "Overhead Press"
        img = "./static/web/Barbell-Standing-Military-Press.gif"
    elif label == "dead":
        prediction = "Deadlift"
        img = "./static/web/Barbell-Deadlift.gif"
    elif label == "rest":
        prediction = "Resting"
        img = "./static/web/exhausted-tired.gif"

    count_rep = False
    count_img = False

    # Count repetitions
    if label != "rest":
        count_rep = tracker_predictor.count_repetitions(label)
        count_img = "./static/pred/count_rep.png"

    return render_template(
        "index.html",
        prediction=prediction,
        exr_img=img,
        count_rep=count_rep,
        count_rep_img=count_img,
    )


if __name__ == "__main__":

    app.run(debug= True , host='0.0.0.0' , port = 5000 )

    app.run()
