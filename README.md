# Real-Time Gender Voice Recognition

Real-Time Gender Voice Recognition is a machine learning project designed to classify gender based on the acoustic properties of voice and speech. It utilizes advanced algorithms and statistical methods to ensure high accuracy and performance in real-time voice file processing. 

## Features

- **Machine Learning Models**: Implemented Logistic Regression, Support Vector Machines (SVM), Random Forest (RF), and XGBoost to classify gender.
- **Feature Selection**: Focused on the top 5 acoustic properties critical for gender recognition.
- **Audio Processing**: Utilized `audio`, `signal`, and `tuneR` libraries in R for effective audio preprocessing.
- **Real-Time Interface**: Developed a Shiny-based user interface for live processing and predictions.
- **Exploratory Data Analysis (EDA)**: Conducted comprehensive analysis to identify and visualize the most significant features contributing to model performance.

## Installation

### Prerequisites

Ensure you have the following installed:
- R
- RStudio
- Required R libraries: `audio`, `signal`, `tuneR`, `shiny`, `caret`, `xgboost`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RealTimeGenderVoiceRecognition.git
   ```
2. Open the project in RStudio.
3. Install required packages:
   ```R
   install.packages(c("audio", "signal", "tuneR", "shiny", "caret", "xgboost"))
   ```
4. Run the Shiny application:
   ```R
   shiny::runApp("app.R")
   ```

## Usage

1. **Data Upload**: Upload a voice recording file (.wav format) via the Shiny interface.
2. **Real-Time Analysis**: The system processes the audio file and predicts the gender based on acoustic properties.
3. **Visualization**: View feature importance and prediction confidence through interactive plots.

## Machine Learning Models

- **Logistic Regression**: For baseline classification.
- **Support Vector Machines (SVM)**: For robust decision boundaries.
- **Random Forest (RF)**: For feature importance and ensemble learning.
- **XGBoost**: For high-performance gradient boosting.

## Libraries Used

- **Audio Processing**: `audio`, `signal`, `tuneR`
- **Machine Learning**: `caret`, `xgboost`
- **UI Development**: `shiny`


## Contact

- **Name**: Samarth Otari
- **GitHub**: https://github.com/samarth49
- **Email**: otarisamarth49@gmail.com
