# BiztelAI DS Assignment 🚀

Welcome to the BiztelAI DS Assignment project! This repository contains production-ready Python code for data processing, exploratory data analysis (EDA), and a REST API built with FastAPI. The project is designed to be modular, scalable, and ready for deployment. 🌟

## Table of Contents
- [Demo](#Demo)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Docker Integration](#docker-integration)
- [Contributing](#contributing)
- [License](#license)

## Demo
![Demo Video](demo.gif)

## Overview 💡
The BiztelAI DS Assignment project demonstrates:
- **Data Ingestion & Preprocessing:** Load and clean a JSON dataset using Pandas, NumPy, and NLTK with an object-oriented pipeline.
- **Advanced EDA:** Generate impressive visualizations (bar plots, boxplots, word clouds, correlation matrices) using Matplotlib, Seaborn, and WordCloud.
- **REST API:** Serve key functionalities (data summary, real-time preprocessing, transcript insights) using FastAPI with an interactive UI.
- **Deployment Ready:** The project is optimized and containerized with Docker for seamless deployment.

## Project Structure 📁
```
BiztelAI_DS_Assignment/
├── api.py                             # FastAPI application (REST API + interactive UI)
├── data_processing.py                 # Data loading, cleaning, and transformation modules
├── eda.py                             # Exploratory Data Analysis functions & visualizations
├── templates/                         # HTML templates for the UI (index.html, eda.html, result.html, dataset_summary.html)
├── BiztelAI_DS_Dataset_Mar'25.json    # Provided dataset file
├── requirements.txt                   # Project dependencies
├── EDA Insights & Methodologies       # The documentation for all insights and methodologies
└── README.md                          # This file
```


## Features ✨
- **Modular Data Pipeline:** Organized into classes (DataLoader, DataCleaner, DataTransformer) for maintainability.
- **Advanced EDA Visualizations:** Interactive and well-organized plots (message counts, sentiment distributions, word clouds, correlation matrix).
- **Interactive API UI:** User-friendly web pages for testing endpoints with forms and buttons.
- **Error Handling & Logging:** Robust error management and logging for production-level code.
- **Dockerized Deployment:** Containerized with Docker for easy deployment to cloud platforms.
- **Asynchronous API Endpoints:** Built with FastAPI for high performance and scalability.

## Installation & Setup 🔧

### Clone the Repository
```git clone https://github.com/Ria2810/BiztelAI_DS_Assignment.git```

### Create and Activate Virtual Environment
```python -m venv venv```

- **On Linux/macOS:**  
```source venv/bin/activate```

- **On Windows:**  
```venv\Scripts\activate```

### Install Dependencies
```
pip install --upgrade pip  
pip install -r requirements.txt
```

### Download NLTK Data (if not already downloaded)
Open a Python shell and run:  
```
import nltk  
nltk.download('punkt')  
nltk.download('stopwords')  
nltk.download('wordnet')
```
## Running the Project ▶️

### Run the API Server
Start the FastAPI server with Uvicorn:  
```uvicorn api:app --reload```

- **Home Page:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the interactive homepage.  
- **EDA Page:** Visit [http://127.0.0.1:8000/eda](http://127.0.0.1:8000/eda) for advanced EDA visualizations.  
- **Dataset Summary:** Visit [http://127.0.0.1:8000/dataset_summary](http://127.0.0.1:8000/dataset_summary) (or your designated endpoint) to view a well-organized dataset summary.

### Testing Endpoints via UI
- **Preprocess Endpoint:** Use the form on the homepage to submit raw text and view its processed result.  
- **Insights Endpoint:** Submit a transcript JSON through the provided form to receive a summary and accompanying graphs.

## API Endpoints 📡
- **GET /**  
  Returns a welcome message and navigation links.

- **GET /dataset_summary**  
  Returns a JSON summary of the dataset (or renders an HTML page if configured).

- **POST /preprocess**  
  Accepts raw text and returns processed text (tokenized, lemmatized, etc.).

- **POST /insights**  
  Accepts a transcript JSON and returns a summary (message counts, sentiments).

- **GET /eda**  
  Renders an HTML page with various EDA visualizations.

- **POST /preprocess_form & /insights_form**  
  HTML form endpoints for user interaction.

## Docker Integration 🐳

### Build the Docker Image
```docker build -t biztelai_ds_assignment .```

### Run the Docker Container
```docker run -d -p 8000:8000 biztelai_ds_assignment```

### Access the Application
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Contributing 🤝
Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements or new features.

## License 📄
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy Coding! 🎉
