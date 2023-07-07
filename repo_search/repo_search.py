print("Loading libraries...")
import openai
import faiss
import tqdm
import os
import sys
import zipfile
import urllib.request
import tiktoken
import datasets
import time

# Module constants

# Map from url to what we need to append to get the zip file
supported_remote_repositories = {
    'https://github.com': "/archive/refs/heads/main.zip",
    'https://gitlab.com': "/-/archive/main/main.zip",
    'https://bitbucket.org': "/get/main.zip",
}

# OpenAI API key
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

OPENAI_MODEL = 'text-embedding-ada-002'

OPENAI_MODEL_MAX_INPUT_TOKENS = 8191

# Exported functions

def generate_embeddings_for_repository(
    dataset_name,
    repo_url_or_path,
    embeddings_dir,
    verbose):
    if dataset_exists(dataset_name, embeddings_dir):
        # To help the user, give the full disk path to the embeddings directory
        embeddings_dir_expanded = os.path.abspath(embeddings_dir)

        print(f'Dataset named {dataset_name} already exists in embeddings directory ({embeddings_dir_expanded}), delete it first if you want to regenerate it.')
        return
    
    # Check if given repository is a URL or a local path
    if repo_url_or_path.startswith('http'):
        # Remote repositories
        if repo_url_or_path.endswith('.zip'):
            generate_embeddings_for_remote_zip_archive(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        elif is_supported_remote_repository(repo_url_or_path):
            generate_embeddings_for_remote_repository_archive(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        else:
            print(f'ERROR: Unsupported remote repository: {repo_url_or_path}')
            return
    else:
        # Local repositories
        if repo_url_or_path.startswith('~'):
            repo_url_or_path = os.path.expanduser(repo_url_or_path)
        
        if os.path.isdir(repo_url_or_path):
            generate_embeddings_for_local_repository(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        elif repo_url_or_path.endswith('.zip'):
            generate_embeddings_for_local_zip_archive(dataset_name, repo_url_or_path, embeddings_dir, verbose)
        else:
            print(f'ERROR: Unsupported local repository: {repo_url_or_path}')
            return



def query_embeddings(
    dataset_name,
    query,
    embeddings_dir,
    verbose):
    print('Querying embeddings...')



# Internal functions
def dataset_exists(dataset_name, embeddings_dir):
    # Check if folder named dataset_name exists in embeddings_dir.
    return os.path.exists(os.path.join(embeddings_dir, dataset_name))

def is_supported_remote_repository(repo_url):
    # Check if the given repo_url is supported by this script.
    for supported_url in supported_remote_repositories:
        if repo_url.startswith(supported_url):
            return True
    return False

def get_download_url_for_remote_repository(repo_url):
    # Get the zip file URL for the given remote repository URL.
    for supported_url in supported_remote_repositories:
        if repo_url.startswith(supported_url):
            return repo_url + supported_remote_repositories[supported_url]
    return None


## Generator functions
def generate_embeddings_for_remote_repository_archive(
    dataset_name,
    repo_url,
    embeddings_dir,
    verbose):
    assert is_supported_remote_repository(repo_url)

    for supported_url in supported_remote_repositories:
        if repo_url.startswith(supported_url):
            download_url = repo_url + supported_remote_repositories[supported_url]
            break
    else:
        print(f'ERROR: Unsupported remote repository: {repo_url}')
        return

    if verbose:
        print(f'Detected {supported_url} repository.')

    generate_embeddings_for_remote_zip_archive(
        dataset_name,
        download_url,
        embeddings_dir,
        verbose
    )


def generate_embeddings_for_remote_zip_archive(
    dataset_name,
    zip_url,
    embeddings_dir,
    verbose):
    if verbose:
        print(f'Downloading {download_url}...')

    # Use zipfile to browse the contents of the zip file without extracting it.
    with urllib.request.urlopen(download_url) as response:
        with zipfile.ZipFile(response) as zip_ref:
            generate_embeddings_for_zipfile(
                dataset_name,
                zip_ref,
                embeddings_dir,
                verbose
            )

def generate_embeddings_for_local_zip_archive(
    dataset_name,
    zip_path,
    embeddings_dir,
    verbose):
    if verbose:
        print(f'Loading {zip_path}...')
    
    # Use zipfile to browse the contents of the zip file without extracting it.
    with zipfile.ZipFile(zip_path) as zip_ref:
        generate_embeddings_for_zipfile(
            dataset_name,
            zip_ref,
            embeddings_dir,
            verbose
        )

def generate_embeddings_for_zipfile(
    dataset_name,
    zipfile,
    embeddings_dir,
    verbose):

    if verbose:
        print(f'Generating embeddings from zipfile for {dataset_name}...')
    
    file_list = zipfile.namelist()

    # For each file in the zip file, generate embeddings for it.
    all_embeddings = []
    for file_path in tqdm.tqdm(file_list):
        with zipfile.open(file_path) as file:
            file_contents = file.read()
            all_embeddings.append(generate_embeddings_for_contents(file_contents))
    
    # Generate a dataset from the embeddings.
    dataset = datasets.Dataset.from_dict({
        'file_path': file_list,
        'embeddings': all_embeddings
    })

    # Save the dataset to disk.
    dataset.save_to_disk(os.path.join(embeddings_dir, dataset_name))

    # Generate index using FAISS.
    generate_faiss_index_for_dataset(dataset, dataset_name, embeddings_dir, verbose)


def generate_embeddings_for_local_repository(
    dataset_name,
    repo_path,
    embeddings_dir,
    verbose):

    if verbose:
        print(f'Generating embeddings from local directory {repo_path} for {dataset_name}...')

    dirs_left = [repo_path]

    file_paths = []
    embeddings = []
    skipped_file_count = 0

    while len(dirs_left) > 0:
        next_dir = dirs_left.pop(0)

        for root, dirs, files in os.walk(next_dir):
            for dir in dirs:
                dirs_left.append(os.path.join(root, dir))
            
            for file in files:
                file_path = os.path.join(root, file)

                # Check if the file can be read as a text file
                try:
                    with open(os.path.join(root, file), 'rt') as file:
                        _ = file.read() # TODO: Is this needed? Does the exception get generated by open or read?
                except UnicodeDecodeError as e:
                    if verbose:
                        print(f'WARNING: Could not read as text file: {file_path}')
                    skipped_file_count += 1
                    continue
                
                # When storing file_path, remove shared repo_path prefix.
                relative_file_path = file_path[len(repo_path) + 1:]
                file_paths.append(relative_file_path)
    
    for file_path in tqdm.tqdm(file_paths):
        full_file_path = os.path.join(repo_path, file_path)
        try:
            with open(full_file_path, 'rt') as file:
                file_contents = file.read()
                embedding = generate_embeddings_for_contents(file_contents, verbose)
                embeddings.append(embedding)
        except:
            if verbose:
                print(f'WARNING: Issue generating embeddings for: {full_file_path}')
            skipped_file_count += 1
            embeddings.append([])
    
    # Generate a dataset from the embeddings.
    dataset = datasets.Dataset.from_dict({
        'file_path': file_paths,
        'embeddings': embeddings
    })

    # Save the dataset to disk.
    dataset.save_to_disk(os.path.join(embeddings_dir, dataset_name))

    # Generate index using FAISS.
    generate_faiss_index_for_dataset(dataset, dataset_name, embeddings_dir, verbose)

def generate_embeddings_for_contents(
    file_contents,
    verbose
):
    # Use tiktoken to split file_contents into chunks of OPENAI_MODEL_MAX_INPUT_TOKENS.
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(file_contents)

    # Split tokens into chunks of OPENAI_MODEL_MAX_INPUT_TOKENS.
    all_embeddings = []
    for i in range(0, len(tokens), OPENAI_MODEL_MAX_INPUT_TOKENS):
        chunk = tokens[i:i + OPENAI_MODEL_MAX_INPUT_TOKENS]
        chunk = encoding.decode(chunk)
        all_embeddings.append(generate_embedding_for_chunk(chunk, verbose))
    
    #print(all_embeddings)
    return all_embeddings

def generate_embedding_for_chunk_FAKE(
    file_chunk,
    verbose
):
    # Debug testing
    return [0.0] * 1536

def generate_embedding_for_chunk(
    file_chunk,
    verbose,
):
    current_try = 0
    max_tries = 3

    while current_try <= max_tries:
        current_try += 1
        try:
            embedding_response = openai.Embedding.create(
                input=file_chunk,
                model=OPENAI_MODEL,
            )
            break
        except openai.error.OpenAIError as e:
            if verbose:
                print(f'WARNING: OpenAI API error: {e}')
            
            # Exponential backoff
            time.sleep(2**current_try)

    return embedding_response['data'][0]['embedding']

def generate_faiss_index_for_dataset(
    dataset,
    dataset_name,
    embeddings_dir,
    verbose):
    pass