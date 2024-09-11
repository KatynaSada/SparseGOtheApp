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
    # The above code is loading a mapping of cell IDs using a function called `load_mapping` and
    # storing the result in a variable called `cell2id_mapping`.
    cell2id_mapping = load_mapping(cell2id)  # Load cell ID mapping
    drugs_data = get_compound_names(compounds_txt)  # Load drug names
    drugs_data.pop(0)  # Remove the first entry if it's a header or unwanted

    return cell_features, drug_features, drug2id_mapping, cell2id_mapping, drugs_data
@st.cache_data
def get_audrc_mean(all_samples_features, drug_features, drug2id_mapping, drugs_data, _model, device):
    # sourcery skip: inline-immediately-returned-variable
    """
    Calculate the AUDRC values for each sample based on their features and drug features,
    store the AUDRC values in a matrix for each drug, and compute the mean per row.

    Parameters:
    - all_samples_features: A list of arrays, each containing features of a sample.
    - drug_features: A list or array of features for each drug.
    - drug2id_mapping: A mapping of drug identifiers.
    - drugs_data: Data containing drug information.
    - model: The model to use for calculating AUDRC.
    - device: The device (CPU or GPU) to perform the calculations.

    Returns:
    - audrc_mean_per_sample: A NumPy array containing the mean AUDRC values per sample.
    """
    num_samples = len(all_samples_features)
    num_drugs = len(drug2id_mapping)
    
    audrc_matrix = np.zeros((num_samples, num_drugs))
    
    for sample_idx, sample_features in enumerate(all_samples_features):
        sample_and_drugs_features = [
            np.concatenate((sample_features, drug_features[i]), axis=None)
            for i in range(num_drugs)
        ]

        sample_and_drugs_features = torch.FloatTensor(np.array(sample_and_drugs_features)).to(device)
        audrc_values = model(sample_and_drugs_features)
        
        audrc_matrix[sample_idx] = audrc_values.detach().numpy().reshape(-1)
    
    # Compute the mean per row (per sample)
    audrc_mean_per_sample = np.mean(audrc_matrix.T, axis=1)
    
    # Create DataFrames for smiles and AUDRC
    df_smiles_names = pd.DataFrame(drugs_data, columns=['Smile', 'Name'])
    df_AUDRC = pd.DataFrame(audrc_mean_per_sample, columns=['AUDRC'])    
    AUDRC_cell = pd.concat( [df_smiles_names[['Name']], df_AUDRC,df_smiles_names[['Smile']]], axis=1).sort_values(by='AUDRC', ascending=True)    
    return AUDRC_cell
def generate_audrc_bar_chart(AUDRC_cell, slider_num):
    """
    Generate a bar chart showing the AUDRC values for the top drugs based on cell-specific features.

    Parameters:
    - AUDRC_cell: DataFrame containing drug names, AUDRC values, and other relevant data.
    - slider_num: Number of top drugs to display in the bar chart (default: 10).
    """
    # Get the top drugs and their AUDRC values
    top_drugs = AUDRC_cell.head(slider_num)
    
    # Create a bar chart using Plotly
    fig = px.bar(top_drugs, x='Name', y='AUDRC',
                 title='AUDRC Values for Most Harmful Drugs',
                 labels={'AUDRC': 'Drug Response (AUDRC)', 'Name': 'Drugs'},
                 color='AUDRC',  # Optional: color bars by AUDRC value
                 color_continuous_scale=px.colors.sequential.Sunset)  # Colorscale for the bars
    
    # Show the Plotly figure in Streamlit
    st.plotly_chart(fig)
def validate_uploaded_file(uploaded_file, example_file):
    try:
        # Check if the uploaded file is a text file
        if not uploaded_file.name.endswith('.txt'):
            st.error("Please upload a .txt file.")
            return False
        
        # Read the example file to get the expected number of values
        with open(example_file, 'r') as file:
            example_values = file.read().strip().split(',')
            expected_count = len(example_values)
        
        # Read the uploaded file
        content = uploaded_file.getvalue().decode("utf-8").strip()
        # Split the content by newline characters to get individual lines
        lines = content.split('\n')
        
        uploaded_samples = []
        for line in lines:
            values = line.split(',')
            array_values = []
            for value in values:
                try:
                    float_value = float(value)
                    array_values.append(float_value)
                except ValueError:
                    st.error("All values must be numerical. A feature with a value of *" + value + "* has been detected.")
                    return False
            uploaded_samples.append(np.array(array_values))
        
        # Validate the number of values in each array
        for sample in uploaded_samples:
            if len(sample) != expected_count:
                st.error(f"Each line in the uploaded file must contain {expected_count} numeric values, but one or more lines have a different number of values.")
                return False

        st.success("File is valid!")
        return True
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False
def update_audrc():
    st.session_state.AUDRC_cell = None  # Reset AUDRC_cell, created because the bar chart stayed there when choosing another data source
@st.cache_data
def get_cell_types(cell2id_mapping):
    """
    Create a dictionary of cell types from a mapping of cell names to IDs.

    Parameters:
    cell2id_mapping (dict): A dictionary where keys are cell names and values are IDs.

    Returns:
    tuple: A dictionary of cell types and a set of unique cancer types.
    """
    # Create a new dictionary to hold cell types
    cell_types = {}
    
    # Populate the new dictionary with cancer types
    for cell_name in cell2id_mapping.keys():
        # Split the cell name by underscore and join everything after the first part
        cancer_type = '_'.join(cell_name.split('_')[1:])  # Join all parts after the first underscore
        
        # Add to the new dictionary
        cell_types[cell_name] = cancer_type
    
    # Get unique cancer types
    unique_cancers = set(cell_types.values())
    
    return cell_types, unique_cancers
    
# Download required data from GitLab
REPO_URL = 'https://gitlab.com/katynasada/sparsego4streamlit.git'  # Replace with your repository URL=?
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
    st.subheader("Use this tool to predict the response of a cell to more than 1500 drugs.")
    st.write("Our neural networks predict a continuous value that represents the area under the dose-response curve (AUDRC) normalized such that **AUDRC = 0 represents complete cell death, AUDRC = 1 represents no effect, and AUDRC > 1 represents that the treatment favours cell growth**.")
    st.write("**Note:** If the predictions are computed for more than one sample, the mean for each drug is calculated.")
    model = st.selectbox('What type of omics data do you want to use?',('Mutations', 'Expression', 'Mutations and expression'))

    if model == "Expression":
        inputdir="sparsego4streamlit_cloned/SparseGO/data/CLs_expression4transfer/allsamples/"
        resultsdir="sparsego4streamlit_cloned/SparseGO/results/CLs_expression4transfer/allsamples/"
        omics_type = "cell2expression"
        cell_features, drug_features, drug2id_mapping, cell2id_mapping, drugs_data = load_all_data(inputdir, resultsdir, omics_type, device, typed="")

        # Load required model
        model = load_model(resultsdir, device)
        
        input_type = st.selectbox('Select your data source for prediction:', ('Upload cells/patients data', 'Use CCLE cell lines'),index=None,on_change=update_audrc)

        if input_type == "Upload cells/patients data":
            gene2id_file = f"{inputdir}gene2ind.txt"
            example_file = f"{inputdir}mycellexpression.txt"
            # Write instructions for the user
            st.write(f"**Please upload the expression data for the {len(pd.read_csv(gene2id_file, sep='\t'))+1} genes in a text file. Each line should represent a sample with expression values for all genes separated by commas.**")
            # Create two columns for the buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download Required Gene List (Must Be in This Order)", open(gene2id_file), file_name="gene2id.txt")
            with col2:
                st.download_button("Download Example Features File", open(example_file), file_name="mycellexpression.txt")

            # File uploader for user to upload their cell features
            uploaded_file = st.file_uploader("Upload Your Cell Features Here")
            if uploaded_file is not None and (validate_uploaded_file(uploaded_file,example_file) and st.button('Predict drug response 💊')):
                content = uploaded_file.getvalue().decode("utf-8").strip()
                # Split the content by newline characters to get individual lines and Split each line by commas to create separate arrays
                lines = content.split('\n')
                uploaded_samples = [np.array([float(value) for value in line.split(',')]) for line in lines]
                st.session_state.AUDRC_cell = get_audrc_mean(uploaded_samples, drug_features, drug2id_mapping, drugs_data, model, device)
            
            if st.session_state.AUDRC_cell is not None:
                st.write(st.session_state.AUDRC_cell)
                slider_num = st.slider("Number of drugs", value=10, max_value=len(drug2id_mapping), key="drug_slider")
                generate_audrc_bar_chart(st.session_state.AUDRC_cell, slider_num)

        elif input_type == "Use CCLE cell lines":
            
            col1, col2 = st.columns(2)
            with col1:
                cell_names = st.multiselect('Select one or more cell lines 🦠',cell2id_mapping, key="multiselect_cells")
            with col2:
                cell_types, unique_cancers = get_cell_types(cell2id_mapping)
                selected_cancer_type = st.multiselect('**and/or** select all cell lines of a cancer type 🧠 🫁 🩸 🦴',unique_cancers, key="multiselect_cancer")
                cell_names_cancer = [cell_name for cell_name, cancer in cell_types.items() if cancer in selected_cancer_type]
            
            # Combine lists without duplicates using set
            all_cell_names = list(set(cell_names) | set(cell_names_cancer))
            st.info(f"You're predicting the drug response of {len(all_cell_names)} cell lines.")
            
            if st.button('Predict drug response 💊'):
                cell_specific_features = []
                for name in all_cell_names:
                    cell_idx = cell2id_mapping.get(name)  # Get the index of the cell from the cell name using a mapping dictionary
                    cell_specific_features.append(cell_features[cell_idx])  # Retrieve the specific features for the cell at the given index
                st.session_state.AUDRC_cell = get_audrc_mean(cell_specific_features, drug_features, drug2id_mapping, drugs_data, model, device)
                
            if st.session_state.AUDRC_cell is not None:
                st.write(st.session_state.AUDRC_cell)
                slider_num = st.slider("Number of drugs", value=10, max_value=len(drug2id_mapping), key="drug_slider")
                generate_audrc_bar_chart(st.session_state.AUDRC_cell, slider_num)
        
            
elif menu =='MoA':
    st.title("Predict the Mechanism of Action")

    
