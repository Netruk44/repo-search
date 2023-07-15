Update command
 - i.e. repo_search update openmw ~/Developer/openmw/apps
 - New column in dataset: file_hash - MD5/SHA1/some kind hash of file contents
 - For every file in the dataset
    - if hash on disk != hash in dataset, update the embedding
    - if file not on disk, remove from dataset
 - For every file in folder, if not in dataset, add it

FAISS integration

Add some kind of a marker to the dataset that indicates which model type and model name created it so the user doesn't need to specify those on the command line when querying