# To Do List
## Up Next / Pending
Update command
 - i.e. repo_search update openmw ~/Developer/openmw/apps
 - New column in dataset: file_hash - MD5/SHA1/some kind hash of file contents
 - For every file in the dataset
    - if hash on disk != hash in dataset, update the embedding
    - if file not on disk, remove from dataset
 - For every file in folder, if not in dataset, add it

Remove/delete command
 - i.e. repo_search remove openmw
 - Remove the dataset and all embeddings for the given repository

List command
 - i.e. repo_search list
 - List all repositories that have been indexed

FAISS integration

## Completed
### v0.3.0
- Add some kind of a marker to the dataset that indicates which model type and model name created it so the user doesn't need to specify those on the command line when querying