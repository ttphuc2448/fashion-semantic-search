# Semantic Fashion Search Engine

A Semantic Text-to-Image search engine built for fashion products. This prototype allows users to search through an image dataset using natural language queries instead of exact keyword matches. It understands concepts like color, patterns, and contexts (e.g., "red summer dress" or "leather shoes").

## 🎬 Demo

[![Watch Demo](https://img.youtube.com/vi/E-gOk5zPwss/maxresdefault.jpg)](https://youtu.be/E-gOk5zPwss)

> 🔗 **Video Demo**: [https://youtu.be/E-gOk5zPwss](https://youtu.be/E-gOk5zPwss)


## How It Works
1. **Offline Encoding:** The AI model processes all local images, extracts visual features (vector embeddings), and saves them to a database.
2. **Online Searching:** When you type a query, the text is converted into a vector. The system calculates the similarity between your text vector and the image vectors, returning the best matches instantly.

## Tech Stack
* **Python**
* **Streamlit** (for the web interface)
* **PyTorch** & **Hugging Face Transformers** (for the OpenAI CLIP model)
* **Pillow** & **Numpy**

---

## 🚀 How to Set Up the Project (For Beginners)

Follow these steps to get this project running on your own machine from scratch!

### 1. Clone the Repository
First, download the project to your computer by cloning it via Git:
```bash
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName
```

### 2. Create a Virtual Environment (Optional but Recommended)
It's a good practice to create a virtual environment to keep your dependencies clean.
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 4. Add Your Images
Before running the engine, you need to provide it with a dataset.
* Create a folder path: `data/images/`
* Drop your `.jpg`, `.jpeg`, or `.png` images into the `data/images/` folder.

### 5. Build the Vector Database
Once your images are in place, run the encoder script. This will download the CLIP model, process your images, and create an `embeddings/` folder.
*(Note: This might take a few minutes depending on how many images you have).*
```bash
python encode_data.py
```

### 6. Start the Application
Once the embeddings are successfully saved, launch the Streamlit search engine:
```bash
python -m streamlit run app.py
```
This will automatically open the web interface in your browser. (If it doesn't open automatically, navigate to [http://localhost:8501](http://localhost:8501)).

*(Watch the [Demo Video](https://youtu.be/E-gOk5zPwss) in action!)*

---

## 🔍 Example Test Queries
Try typing these into the search bar to test the semantic understanding:
* "A red summer dress"
* "Striped button-up shirt"
* "Formal business attire"
* "Something to wear to the beach"
