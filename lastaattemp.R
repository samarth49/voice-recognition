library(shiny)
library(xgboost)
library(GGally)
library(dplyr)
library(egg)
library(caret)
library(ggplot2)
library(randomForest)
library(e1071)
library(wrassp)
library(readr)
library(tuneR)
library(signal)
library(oce)
library(audio)

voice <- read.csv('voice.csv')

# Data preprocessing
voice$label <- ifelse(voice$label == "male", 1, 0)
voice <- voice %>% mutate(label = as.factor(label))

# Train-test split
set.seed(123)
index <- createDataPartition(voice$label, p = 0.8, list = FALSE)
train_data <- voice[index,]
test_data <- voice[-index,]

# Convert target variable to numeric (required for XGBoost)
train_data$label <- as.numeric(as.character(train_data$label))
test_data$label <- as.numeric(as.character(test_data$label))

# Train XGBoost model
train_matrix <- as.matrix(train_data[, c("meanfun", "sp.ent", "IQR", "Q25", "sd")])
label_vector <- train_data$label
xgb_model <- xgboost(data = train_matrix,
                     label = label_vector,
                     objective = "binary:logistic",
                     nrounds = 20,
                     eta = 0.3,
                     max_depth = 6)

# Feature extraction function
extract_features <- function(audio) {
  shannon_entropy <- function(x) {
    probs <- table(x) / length(x)
    -sum(probs * log2(probs))
  }
  
  mfcc_features <- melfcc(audio,
                          sr = audio@samp.rate,
                          wintime = 0.015,
                          hoptime = 0.005,
                          sumpower = TRUE,
                          nbands = 40,
                          bwidth = 1,
                          preemph = 0.95)
  
  mfcc_coefficients <- mfcc_features
  
  meanfun <- apply(mfcc_coefficients, 2, mean, na.rm = TRUE)
  sp.ent <- apply(mfcc_coefficients, 2, function(x) shannon_entropy(x))
  IQR <- apply(mfcc_coefficients, 2, IQR, na.rm = TRUE)
  Q25 <- apply(mfcc_coefficients, 2, quantile, probs = 0.25, na.rm = TRUE)
  sd_val <- apply(mfcc_coefficients, 2, sd, na.rm = TRUE)
  
  features <- data.frame(meanfun = meanfun, sp.ent = sp.ent, IQR = IQR, Q25 = Q25, sd = sd_val)
  features_matrix <- as.matrix(features)
  
  return(features_matrix)
}

# Shiny app
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body {
        background-color: #f8f9fa;
        color: #212529;
      }
      .sidebar {
        background-color: #343a40;
        border-right: 1px solid #495057;
      }
      .main-panel {
        background-color: #ffffff;
      }
      .title-panel {
        background-color: #343a40;
        border-bottom: 2px solid #495057;
        color: #ffffff;
      }
      .alert-info {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
      }
      #prediction {
        font-size: 18px;
        font-weight: bold;
        color: #007bff;
      }
      .center {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        text-align: center;
      }
      .emojis {
        font-size: 24px;
        margin-top: 20px;
      }
    "))
  ),
  titlePanel("🎤 Voice Gender Recognition 🧑‍🎤", windowTitle = "Voice Gender Recognition"),
  sidebarLayout(
    sidebarPanel(
      div(class = "center",
          actionButton("record_button", "Record Voice 🎤"),
          tags$br(),
          fileInput("upload_file", "Upload Audio File", accept = c(".wav"))
      )
    ),
    mainPanel(
      div(class = "center",
          h3("🔮 Prediction 🔮"),
          hr(),
          verbatimTextOutput("start_speaking"),
          verbatimTextOutput("recording_message"),
          verbatimTextOutput("uploading_message"),
          verbatimTextOutput("prediction_output"),
          verbatimTextOutput("prediction_gender"), # Add this line
          
          tags$br()
      )
    )
  )
)

server <- function(input, output, session) {
  save_path <- reactiveVal(NULL)
  
  observeEvent(input$record_button, {

    rec_time <- 10
    Samples <- rep(NA_real_, 44100 * rec_time)
    cat("Start speaking\n")

    audio_obj <- record(Samples, 44100, 1)
    wait(6)
    rec <- audio_obj$data
    file.create("exam.wav")
    save_path("D:/VIT22-26/SY/DATA SCIENCE/DS LAB/CP/archive/exam.wav")
    save.wave(rec, save_path())
    showNotification("Sound recorded successfully!", duration = 5, type = "message")
    output$recording_message <- renderText("Sound recorded successfully!")
    
    file_path <- "D:/VIT22-26/SY/DATA SCIENCE/DS LAB/CP/archive/exam.wav"
    recorded_audio <- readWave(file_path)
    features <- extract_features(recorded_audio)
    test_mat <- matrix(c(features[1, ]/c(1000, 10, 300, 100, 250)), nrow = 1, ncol = 5, byrow = TRUE)
    predicted_value <- predict(xgb_model, test_mat, type = "response")
    output$prediction_output <- renderText(paste("Predicted value:", predicted_value))
    if (predicted_value > 0.5) {
      output$prediction_gender <- renderText("Predicted gender: Male")
    } else {
      output$prediction_gender <- renderText("Predicted gender: Female")
    }
  })
  
  observeEvent(input$upload_file, {
    file.copy(input$upload_file$datapath, "D:/VIT22-26/SY/DATA SCIENCE/DS LAB/CP/archive/exam.wav", overwrite = TRUE)
    showNotification("File uploaded successfully!", duration = 5, type = "message")
    output$uploading_message <- renderText("File uploaded successfully!")
    
    audio_file <- input$upload_file$datapath
    audio <- readWave(audio_file)
    features <- extract_features(audio)
    test_mat <- matrix(c(features[1, ]/c(1000, 10, 300, 100, 250)), nrow = 1, ncol = 5, byrow = TRUE)
    predicted_value <- predict(xgb_model, test_mat, type = "response")
    output$prediction_output <- renderText(paste("Predicted value:", predicted_value))
    if (predicted_value > 0.5) {
      output$prediction_gender <- renderText("Predicted gender: Male")
    } else {
      output$prediction_gender <- renderText("Predicted gender: Female")
    }
  })
}
shinyApp(ui = ui, server = server)


