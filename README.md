# Advanced Document Similarity Finder

## Purpose

The Advanced Document Similarity Finder is a Streamlit-based web application that allows users to explore document similarities using advanced natural language processing techniques. The app provides the following features:

1. Upload a CSV file containing document descriptions and metadata.
2. Filter documents based on various metadata fields.
3. Choose between document-level and sentence-level embedding strategies.
4. Visualize document similarities using t-SNE or UMAP dimensionality reduction techniques.
5. Search for similar documents by either selecting an existing document or entering a custom description.
6. Display detailed information about similar documents.

This tool is particularly useful for researchers, data analysts, and anyone working with large collections of textual data who need to quickly identify and explore similarities between documents.

## Installation

### Prerequisites

- Anaconda or Miniconda installed on your system

### Setting up the Conda Environment

1. Clone this repository or download the `environment.yml` file.

2. Open a terminal or command prompt and navigate to the directory containing the `environment.yml` file.

3. Create the conda environment by running:
   ```
   conda env create -f environment.yml
   ```

4. Activate the newly created environment:
   ```
   conda activate vector_embeddings_app
   ```

## Running the App

1. Ensure you have activated the `vector_embeddings_app` conda environment.

2. Set up your Nomic API token:
   - Create a `.env` file in the same directory as your `app.py` file.
   - Add your Nomic API token to the `.env` file like this:
     ```
     NOMIC_API_TOKEN=your_api_token_here
     ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Your default web browser should open automatically and display the app. If it doesn't, you can manually open a web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

## Using the App

1. Upload your CSV file containing document descriptions and metadata.
2. Use the sidebar to filter documents based on various fields.
3. Choose your embedding strategy and visualization method.
4. Select an existing document or enter a custom description to find similar documents.
5. Adjust the number of similar documents to display.
6. Explore the visualizations and detailed information about similar documents.

Enjoy exploring your document similarities!