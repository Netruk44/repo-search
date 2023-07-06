import click

@click.group()
def main():
    pass

# 'generate' command, generates a new embeddings dataset.
# Takes in a dataset name and a URL to a git repository.
@main.command(help = 'Generate a new embeddings dataset for a given repository.')
@click.argument('dataset_name')
@click.argument('repo_url_or_path')
@click.option('--embeddings_dir', '-d', default = 'embeddings', help = 'Output directory for the generated embeddings.')
@click.option('--verbose', '-v', is_flag = True, help = 'Print verbose output.')
def generate(
    dataset_name,
    repo_url_or_path, 
    embeddings_dir, 
    verbose):

    from repo_search import generate_embeddings_for_repository

    generate_embeddings_for_repository(dataset_name, repo_url_or_path, embeddings_dir, verbose)

# 'query' command, queries a dataset for a given query.
# Takes in a dataset name and a query string.
@main.command(help = 'Query a dataset using natural language.')
@click.argument('dataset_name')
@click.argument('query')
@click.option('--embeddings_dir', '-d', default = 'embeddings', help = 'Directory containing the embeddings for the dataset.')
@click.option('--verbose', '-v', is_flag = True, help = 'Print verbose output.')
def query(
    dataset_name,
    query,
    embeddings_dir,
    verbose):

    from repo_search import query_embeddings

    query_embeddings(dataset_name, query, embeddings_dir, verbose)

if __name__ == '__main__':
    main()