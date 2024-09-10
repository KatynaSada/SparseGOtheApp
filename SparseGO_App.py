from pathlib import Path
import streamlit
import streamlit as st
from st_files_connection import FilesConnection
from streamlit_option_menu import option_menu
import numpy as np
import torch
import sys # we require code from other folders
import os
import plotly.express as px
import shutil
import pandas as pd
import subprocess
from io import StringIO



device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

@st.cache_data
def clone_and_extract_folder(repo_url, branch_name):
    """
    Clone a GitLab repository into a new directory in the current path and extract a specific folder.
    Then delete the .git directory and .gitattributes file to keep only the folder contents.

    Parameters:
    - repo_url (str): The URL of the GitLab repository to clone.
    - branch_name (str): The branch of the repository to clone.
    """
    
    # Set original_dir to the current working directory
    original_dir = os.getcwd()  # Store the original directory

    # Extract the repository name from the URL
    repo_name = repo_url.split('/')[-1].replace('.git', '')  # e.g., 'REPOSITORY'

    # Create a new directory for the repository
    repo_path = os.path.join(original_dir, f"{repo_name}_cloned")  # Example: 'REPOSITORY_cloned'
    
    # Check if the cloned folder exists and is not empty
    if os.path.exists(repo_path) and os.listdir(repo_path):
        print(f"The cloned folder already exists and is not empty: {repo_path}. Exiting the function.")
        return  # Exit the function
    
    # Ensure Git LFS is initialized
    print("Initializing Git LFS...")
    subprocess.run(["git", "lfs", "install"], check=True)

    # Clone the repository into the new directory
    print(f"Cloning repository from {repo_url} into {repo_path}...")
    subprocess.run(["git", "clone", "--branch", branch_name, repo_url, repo_path], check=True)

    # Change to the cloned repository directory
    os.chdir(repo_path)

    # Pull LFS files
    print("Pulling LFS files...")
    subprocess.run(["git", "lfs", "pull"], check=True)

    # Optionally, you can remove .git and .gitattributes if needed
    print("Cleaning up unnecessary files...")
    shutil.rmtree(os.path.join(repo_path, ".git"))
    os.remove(os.path.join(repo_path, ".gitattributes"))

    print("Repository cloned and cleaned up successfully.")
    
    # Change back to the original directory
    os.chdir(original_dir)
    print(f"Returned to the original directory: {original_dir}")
@st.cache_resource
def load_model(resultsdir, device):
    """
    Load a PyTorch model from a specified directory and set it to evaluation mode.

    Parameters:
    - resultsdir: Directory containing the model file.
    - device: The device (CPU or GPU) on which to load the model.

    Returns:
    - model: The loaded PyTorch model in evaluation mode.
    """
    # Construct the file path for the model
    load_path = f"{resultsdir}last_model.pt"

    # Load the model
    model = torch.load(load_path, map_location=device,weights_only=False)
        
    # If the model is a DataParallel model, extract the original model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        
    return model.eval()  # Set the model to evaluation mode
@st.cache_data
def load_all_data(inputdir, resultsdir, omics_type, device, typed=""):
    """
    Load cell and drug data from specified directories and prepare features and mappings,
    along with loading the model for inference.

    Parameters:
    - inputdir: Directory containing input data files.
    - resultsdir: Directory containing results files.
    - omics_type: mutations, expression, multiomics
    - device: Optional device on which to load the model (e.g., 'cuda:0' or 'cpu').
    - typed: Optional string to modify file names (default is an empty string).

    Returns:
    - cell_features: NumPy array of cell expression features.
    - drug_features: NumPy array of drug fingerprint features.
    - drug2id_mapping: Dictionary mapping drug identifiers.
    - cell2id_mapping: Dictionary mapping cell identifiers.
    - drugs_data: List of drug names.
    - device: The device (CPU or GPU) on which the model is loaded.
    """
    # Construct file paths based on the input directory, results directory, and typed string
    cell2id = f"{inputdir}cell2ind{typed}.txt"
    drug2id = f"{inputdir}drug2ind{typed}.txt"
    drug2fingerprint = f"{inputdir}drug2fingerprint{typed}.txt"
    genotype = f"{inputdir}{omics_type}{typed}.txt"
    compounds_txt = f"{inputdir}compound_names{typed}.txt"

    # Load features and mappings
    cell_features = np.genfromtxt(genotype, delimiter=',')  # Load cell features
    drug_features = np.genfromtxt(drug2fingerprint, delimiter=',')  # Load drug features
    drug2id_mapping = load_mapping(drug2id)  # Load drug ID mapping
    cell2id_mapping = load_mapping(cell2id)  # Load cell ID mapping
    drugs_data = get_compound_names(compounds_txt)  # Load drug names
    drugs_data.pop(0)  # Remove the first entry if it's a header or unwanted

    return cell_features, drug_features, drug2id_mapping, cell2id_mapping, drugs_data
@st.cache_data
def get_audrc_for_cell(cell_name, cell2id_mapping, cell_features, drug_features, drug2id_mapping, drugs_data, _model, device):
    """
    Calculate the AUDRC values for a specific cell based on its features and drug features,
    and create a DataFrame of the results sorted by AUDRC.

    Parameters:
    - cell_name: The name of the cell for which to calculate AUDRC.
    - cell2id_mapping: A dictionary mapping cell names to their indices.
    - cell_features: An array or list containing features for each cell.
    - drug_features: A list or array of features for each drug.
    - drug2id_mapping: A mapping of drug identifiers.
    - model: The model to use for calculating AUDRC.
    - device: The device (CPU or GPU) to perform the calculations.

    Returns:
    - df_combined: A DataFrame containing drug smiles, names, and their corresponding AUDRC values, sorted by AUDRC.
    """
    # Get the index of the cell from the cell name using a mapping dictionary
    cell_idx = cell2id_mapping[cell_name]

    # Retrieve the specific features for the cell at the given index
    cell_specific_features = cell_features[cell_idx]
    
    # Create a list of concatenated features for each drug
    cell_specific_features_drugs = [
        np.concatenate((cell_specific_features, drug_features[i]), axis=None)  # Concatenate cell features with drug features
        for i in range(len(drug2id_mapping))  # Iterate over each drug in the mapping
    ]

    # Convert the list of concatenated features into a PyTorch FloatTensor and move it to the specified device (CPU/GPU)
    cell_specific_features_drugs = torch.FloatTensor(np.array(cell_specific_features_drugs)).to(device)

    # Pass the concatenated features through the model to get the AUDRC values
    AUDRC = model(cell_specific_features_drugs)

    # Create DataFrames for smiles and AUDRC
    df_smiles_names = pd.DataFrame(drugs_data, columns=['Smile', 'Name'])
    df_AUDRC = pd.DataFrame(AUDRC.detach().numpy(), columns=['AUDRC'])

    return pd.concat(
        [df_smiles_names[['Name']], df_AUDRC,df_smiles_names[['Smile']]], axis=1
    ).sort_values(by='AUDRC', ascending=True)
def validate_uploaded_file(uploaded_file, example_file):
    # Check if the uploaded file is a text files  
    if not uploaded_file.name.endswith('.txt'):
        st.error("Please upload a .txt file.")
        return False
    
    # Read the example file to get the expected number of values
    with open(example_file, 'r') as file:
        example_values = file.read().strip().split(',')
        expected_count = len(example_values)
            
    # Read the uploaded file
    content = (uploaded_file.getvalue().decode("utf-8").strip())
    uploaded_values = content.split(',')
    # Validate the number of values
    if len(uploaded_values) != expected_count:
        st.error(f"The uploaded file must contain {expected_count} values, but it contains {len(uploaded_values)}.")
        return False

    # Validate that all values are numerical
    try:
        uploaded_values = [float(value) for value in uploaded_values]
    except ValueError:
        st.error("All values must be numerical.")
        return False

    st.success("File is valid!")
    return True
# Download required data from GitLab
REPO_URL = 'https://gitlab.com/katynasada/sparsego4streamlit.git'  # Replace with your repository URL
BRANCH_NAME = 'main'  # Replace with the branch name
clone_and_extract_folder(REPO_URL, BRANCH_NAME)

with st.sidebar:
    menu = option_menu(None, ["About Us", "Drug Response", "MoA"], 
        icons=['house', 'capsule-pill', "clipboard-data"], # bullseye clipboard-heart joystick https://icons.getbootstrap.com/
        menu_icon="cast", default_index=0, orientation="vertical")

sys.path.append("sparsego4streamlit_cloned/SparseGO/code")
import util
from util import *

if menu =='About Us':
    st.title("About Us")
    st.image("app_elements/logo.png")
    st.write(
        'Welcome to our app! Our goal is to make the findings from [**Discovering the mechanism of action of drugs with a sparse explainable network**](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00333-X/fulltext) accessible to all. Our deep learning models predict cancer drug responses and uses explainable AI to uncover the mechanisms of action behind these drugs, it can also extract the key genomic features for drug response. '
    )
    st.write("We hope this tool is valuable for your work. Feel free to reach out with any interesting discoveries, questions, or suggestions at ksada@unav.es!")
    st.image("app_elements/network.png")
    
elif menu =='Drug Response':
    st.title("Cancer Drug Response Prediction")
    st.write("Use this tool to predict the response of a cell to more than 1500 drugs.")
    st.write("Our neural networks predict a continuous value that represents the area under the dose-response curve (AUDRC) normalized such that **AUDRC = 0 represents complete cell death, AUDRC = 1 represents no effect, and AUDRC > 1 represents that the treatment favours cell growth**.")
    model = st.selectbox('What type of omics data do you want to use?',('Mutations', 'Expression', 'Mutations and expression'))

    if model == "Expression":
        inputdir="sparsego4streamlit_cloned/SparseGO/data/CLs_expression4transfer/allsamples/"
        resultsdir="sparsego4streamlit_cloned/SparseGO/results/CLs_expression4transfer/allsamples/"
        omics_type = "cell2expression"
        # Load required model
        model = load_model(resultsdir, device)
        
        input_type = st.selectbox('Select your data source for prediction:', ('Upload cell/patient data', 'Use CCLE cell line'),index=None)

        if input_type == "Upload cell/patient data":
            gene2id_file = f"{inputdir}gene2ind.txt"
            example_file = f"{inputdir}mycellexpression.txt"
            # Write instructions for the user
            st.write("**Please upload the expression data for the 14,834 genes from your sample in a text file with values separated by commas.**")
            # Create two columns for the buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download Required Gene List (Must Be in This Order)", open(gene2id_file), file_name="gene2id.txt")
            with col2:
                st.download_button("Download Example Features File", open(example_file), file_name="mycellexpression.txt")

            # File uploader for user to upload their cell features
            uploaded_file = st.file_uploader("Upload Your Cell Features Here")
            if uploaded_file is not None:
                validate_uploaded_file(uploaded_file,example_file)

        elif input_type == "Use CCLE cell line":
            cell_features, drug_features, drug2id_mapping, cell2id_mapping, drugs_data = load_all_data(inputdir, resultsdir, omics_type, device, typed="")
            cell_name = st.selectbox('Select cell line',cell2id_mapping,index=None)
            if cell_name is not None:
                AUDRC_cell = get_audrc_for_cell(cell_name, cell2id_mapping, cell_features, drug_features, drug2id_mapping, drugs_data, model, device)
                slider_num = st.slider("Number of drugs", value=15,max_value=len(drug2id_mapping))
                # Get the first 10 drugs and their AUDRC values
                top_drugs = AUDRC_cell.head(slider_num)
                # Create a bar chart using Plotly
                fig = px.bar(top_drugs, x='Name', y='AUDRC', 
                    title='AUDRC Values for Top 10 Drugs',
                    labels={'AUDRC': 'AUDRC', 'Name': 'Drugs'},
                    color='AUDRC',  # Optional: color bars by AUDRC value
                    color_continuous_scale=px.colors.sequential.Sunset)  # https://plotly.com/python/builtin-colorscales/ colors
                # Show the Plotly figure in Streamlit
                st.plotly_chart(fig)
            
            
elif menu =='MoA':
    st.title("Predict the Mechanism of Action")

    
