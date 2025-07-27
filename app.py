import streamlit as st
import os

# Set Streamlit server configuration
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

# Set the default theme to dark mode
st.set_page_config(
    layout="centered",
    initial_sidebar_state="auto",
    page_title="🏋️ Fitness Tracker App 🏋️",
    page_icon="🏋️",
)

st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #0e1117;
    }
    .css-1dp5vir {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from src.models.predict_model import FitnessTrackerPredictor

# Function to load model and perform prediction
def predict_activity(acc_path, gyr_path, model_path, cluster_model_path):
    # Load predictor
    tracker_predictor = FitnessTrackerPredictor(acc_path, gyr_path, model_path, cluster_model_path)
    
    # Predict activity
    label = tracker_predictor.predict_activity()
    
    # Count repetitions
    repetition_count = tracker_predictor.count_repetitions(label)
    
    return label, repetition_count

# Streamlit app  
def main(): 
    st.title("🏋️ Fitness Tracker App 🏋️")
    st.markdown("---")

    # File upload
    st.sidebar.title("📁 Upload Files")
    acc_file = st.sidebar.file_uploader("Upload Accelerometer CSV file:", type=["csv"])
    gyr_file = st.sidebar.file_uploader("Upload Gyroscope CSV file:", type=["csv"])
    st.sidebar.markdown("---")

    # Model paths
    model_path = 'models/final_model.pkl'
    cluster_model_path = 'models/Clustering_model.pkl'

    if acc_file is not None and gyr_file is not None:
        # Button to trigger prediction
        if st.sidebar.button("🚀 Predict"):
            # Perform prediction
            acc_path = 'data/raw/MetaMotion/' + acc_file.name
            gyr_path = 'data/raw/MetaMotion/' + gyr_file.name
            
            label, repetition_count = predict_activity(acc_path, gyr_path, model_path, cluster_model_path)
            
            # Display results
            st.subheader("🔮 Prediction Results")
            st.write("Predicted Activity:", label)
            st.write("Repetition Count:", repetition_count)
            
            # Display corresponding GIF
            if label in ['bench', 'dead', 'ohp', 'squat']:
                st.subheader("🎥 Activity Demo")
                gif_url = {
                    'bench': "https://media0.giphy.com/media/QvXVzMT3oziRDud6df/giphy.gif?cid=6c09b952t6mmzaj7khyy3ktbi3z8vag08osrgrzwtqkbkhae&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=s",
                    'dead': "https://www.journalmenu.com/wp-content/uploads/2017/10/Deadlift-gif-front-view.gif",
                    'ohp': "https://newlife.com.cy/wp-content/uploads/2019/12/00911301-Barbell-Seated-Overhead-Press_Shoulders_360.gif",
               
                }
                st.image(gif_url[label], use_column_width=True)
                st.subheader("Count repetitions")
                st.image('static/pred/count_rep.jpg', use_column_width=True)

            # Show uploaded files
            st.subheader("📂 Uploaded Files")
            st.write("Accelerometer File:", acc_file.name)
            st.write("Gyroscope File:", gyr_file.name)
            
            st.markdown("---")
            
           

    else:
        # Instructions
        st.write("Please upload both Accelerometer and Gyroscope CSV files to predict activity.")

if __name__ == "__main__":
    main()