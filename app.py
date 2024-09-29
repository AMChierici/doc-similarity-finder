import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from nomic import embed
# from nomic import atlas
import nomic
import nltk
import os
from dotenv import load_dotenv
import umap

# # Load environment variables
# load_dotenv()

# # Get Nomic API token
# NOMIC_API_TOKEN = os.getenv('NOMIC_API_TOKEN')

NOMIC_API_TOKEN = st.secrets["NOMIC_API_TOKEN"]

# Configure Nomic with API token
nomic.login(NOMIC_API_TOKEN)

# Function to download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        st.success("NLTK 'punkt_tab' tokenizer is already downloaded.")
    except LookupError:
        st.info("Downloading NLTK 'punkt_tab' tokenizer...")
        nltk.download('punkt_tab')
        st.success("NLTK 'punkt_tab' tokenizer has been downloaded successfully.")

# Call the function to download NLTK data
download_nltk_data()

# Load and preprocess the data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Create embeddings using Nomic
@st.cache_resource
def create_embeddings(texts, strategy):
    if strategy == "document":
        embedding_response = embed.text(texts)
        embeddings = np.array(embedding_response['embeddings'])
    elif strategy == "sentence":
        all_sentences = [nltk.sent_tokenize(doc) for doc in texts]
        flat_sentences = [sentence for doc in all_sentences for sentence in doc]
        sentence_embedding_response = embed.text(flat_sentences)
        sentence_embeddings = np.array(sentence_embedding_response['embeddings'])
        # Aggregate sentence embeddings to document level
        doc_embeddings = []
        idx = 0
        for doc_sentences in all_sentences:
            doc_emb = np.mean(sentence_embeddings[idx:idx+len(doc_sentences)], axis=0)
            doc_embeddings.append(doc_emb)
            idx += len(doc_sentences)
        embeddings = np.array(doc_embeddings)
    return embeddings

# Find similar documents
def find_similar_documents(embeddings, query_embedding, top_n):
    cosine_similarities = cosine_similarity([query_embedding], embeddings).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1][:top_n]
    return related_docs_indices, cosine_similarities[related_docs_indices]

# Dimensionality reduction
@st.cache_resource
def reduce_dimensions(embeddings, method='t-SNE'):
    if method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)

# Create interactive plot
def create_plot(df, reduced_embeddings, selected_index, similar_indices, similarities, method):
    trace_all = go.Scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.5),
        text=df.index,
        hoverinfo='text',
        name='All Documents'
    )
    
    trace_selected = go.Scatter(
        x=[reduced_embeddings[selected_index, 0]] if selected_index is not None else [],
        y=[reduced_embeddings[selected_index, 1]] if selected_index is not None else [],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        text=['Selected Document'],
        hoverinfo='text',
        name='Selected Document'
    )
    
    trace_similar = go.Scatter(
        x=reduced_embeddings[similar_indices, 0],
        y=reduced_embeddings[similar_indices, 1],
        mode='markers',
        marker=dict(size=12, color='green', opacity=0.8),
        text=[f"Similarity: {sim:.2f}" for sim in similarities],
        hoverinfo='text',
        name='Similar Documents'
    )
    
    layout = go.Layout(
        title=f'Document Similarity Visualization ({method})',
        hovermode='closest',
        xaxis=dict(title=f'{method} Component 1'),
        yaxis=dict(title=f'{method} Component 2')
    )
    
    fig = go.Figure(data=[trace_all, trace_selected, trace_similar], layout=layout)
    return fig

# Display document details
def display_document_details(df, index, similarity=None):
    st.write("---")
    if similarity is not None:
        st.write(f"**Similarity: {similarity:.2f}**")
    for column in df.columns:
        st.write(f"**{column}:** {df.iloc[index][column]}")

# Streamlit app
def main():
    st.title("Advanced Document Similarity Finder")
    
    if not NOMIC_API_TOKEN:
        st.error("Nomic API Token not found. Please check your .env file.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Sidebar for filtering
        st.sidebar.title("Filter Documents")
        filter_columns = ['Category', 'level', 'Risk category', 'Risk subcategory', 'Entity', 'Intent', 'Timing', 'Domain', 'Sub-domain']
        filters = {}
        for column in filter_columns:
            if column in df.columns:
                unique_values = df[column].unique()
                selected_values = st.sidebar.multiselect(f"Select {column}", unique_values)
                if selected_values:
                    filters[column] = selected_values
        
        # Apply filters
        mask = pd.Series(True, index=df.index)
        for column, values in filters.items():
            mask &= df[column].isin(values)
        filtered_df = df[mask]
        
        # User inputs
        embedding_strategy = st.selectbox("Select embedding strategy:", ["document", "sentence"])
        visualization_method = st.selectbox("Select visualization method:", ["t-SNE", "UMAP"])
        
        # Create embeddings
        embeddings = create_embeddings(filtered_df['Description'].tolist(), embedding_strategy)
        st.write(f"Embeddings created. Shape: {embeddings.shape}")
        
        # Reduce dimensions for visualization
        reduced_embeddings = reduce_dimensions(embeddings, method=visualization_method)
        
        # Manual input or select existing document
        search_option = st.radio("Choose search option:", ["Select Existing Document", "Manual Input"])
        
        if search_option == "Select Existing Document":
            selected_doc = st.selectbox("Select a document:", filtered_df['Description'])
            selected_index = filtered_df[filtered_df['Description'] == selected_doc].index[0]
            query_embedding = embeddings[selected_index]
        else:
            user_input = st.text_area("Enter your description:", "")
            if user_input:
                query_embedding = create_embeddings([user_input], embedding_strategy)[0]
                selected_index = None
            else:
                st.warning("Please enter a description for manual search.")
                return
        
        top_n = st.slider("Number of similar documents to find:", 1, 10, 5)
        
        similar_indices, similarities = find_similar_documents(embeddings, query_embedding, top_n)
        
        if len(similar_indices) > 0:
            # Create and display the plot
            fig = create_plot(filtered_df, reduced_embeddings, selected_index, similar_indices, similarities, visualization_method)
            st.plotly_chart(fig)
            
            # Display selected document details
            if search_option == "Select Existing Document":
                st.subheader("Selected Document:")
                display_document_details(filtered_df, selected_index)
            else:
                st.subheader("Query:")
                st.write(user_input)
            
            # Display similar documents
            st.subheader("Similar Documents:")
            for idx, sim in zip(similar_indices, similarities):
                display_document_details(filtered_df, idx, sim)
        else:
            st.warning("No similar documents found. This could be due to an error in processing.")

if __name__ == "__main__":
    main()