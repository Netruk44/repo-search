import click
import logging
import os

@click.group()
def main():
    pass


COLOR_DARK_GREY = "\x1b[38;5;240m"
COLOR_BOLD = "\x1b[1m"
COLOR_RESET = "\x1b[0m"
LOGGING_FORMAT = COLOR_DARK_GREY + '[%(asctime)s - %(name)s]' + COLOR_RESET + COLOR_BOLD + ' %(levelname)s:' + COLOR_RESET + ' %(message)s'

# 'generate' command, generates a new embeddings dataset.
# Takes in a dataset name and a URL to a git repository.
@main.command(help = 'Generate a new embeddings dataset for a given repository.')
@click.argument('dataset_name')
@click.argument('repo_url_or_path')
@click.option('--model_type', '-m', default = 'instructor', help = 'Which model to use for generating the embeddings. Options: instructor, openai. Default: instructor')
@click.option('--model_name', '-n', default = None, help = 'Which model to use for generating the embeddings. Options available depend on the model type.')
@click.option('--embeddings_dir', '-d', default = None, help = 'Output directory for the generated embeddings.')
@click.option('--verbose', '-v', is_flag = True, help = 'Print verbose output.')
def generate(
    dataset_name,
    repo_url_or_path,
    model_type,
    model_name,
    embeddings_dir, 
    verbose):

    from repo_search import generate_embeddings_for_repository
    
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG if verbose else logging.INFO)

    if embeddings_dir is None:
        embeddings_dir = get_default_embeddings_dir()

    generate_embeddings_for_repository(dataset_name, repo_url_or_path, embeddings_dir, model_type, model_name)

# 'query' command, queries a dataset for a given query.
# Takes in a dataset name and a query string.
@main.command(help = 'Query a dataset using natural language.')
@click.argument('dataset_name')
@click.argument('query')
@click.option('--show', '-s', default = 't10', help = 'Which results to show. Prefix t for top, b for bottom. Suffix with % to show a percentage of results. t10=top 10, b5%=bottom 5%')
@click.option('--embeddings_dir', '-d', default = None, help = 'Directory containing the embeddings for the dataset.')
@click.option('--verbose', '-v', is_flag = True, help = 'Print verbose output.')
def query(
    dataset_name,
    query,
    show,
    embeddings_dir,
    verbose):

    from repo_search import query_embeddings

    logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG if verbose else logging.INFO)

    if embeddings_dir is None:
        embeddings_dir = get_default_embeddings_dir()

    show_func = show_str_to_function(show)

    # Make some space between the progress output and the results.
    results = show_func(*query_embeddings(dataset_name, query, embeddings_dir))

    print()
    print()
    print()
    print(f'"{query}"')
    print()

    for similarity, estimated_location, file_path in results:
        similarity *= 100
        print(f'{similarity:.2f}% match\t{file_path} [{estimated_location} of the way through]')


def get_default_embeddings_dir():
    # Return the 'embeddings' directory in the same directory as this file.
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'embeddings')

# 'Show' logic

# Order and slice function generators.
# These functions return a function that will be used to order and slice the complete result list.
def show_n(n, reverse = False):
    return lambda similarities, estimated_location, file_paths: sorted(zip(similarities, estimated_location, file_paths), reverse = reverse)[:n]

def show_n_percent(n, reverse = False):
    return lambda similarities, estimated_location, file_paths: sorted(zip(similarities, estimated_location, file_paths), reverse = reverse)[:int(len(similarities) * n / 100)]

def show_str_to_function(show_str):
    # t10 -> show_top_n(10)
    # b10 -> show_bottom_n(10)
    # t10% -> show_top_n_percent(10)
    # b10% -> show_bottom_n_percent(10)

    assert show_str.startswith('t') or show_str.startswith('b'), 'Show string must start with t(op) or b(ottom).'
    show_top = show_str.startswith('t')
    show_str = show_str[1:]

    if show_str.endswith('%'):
        show_str = show_str[:-1]
        show_func = show_n_percent(int(show_str), reverse=show_top)
    else:
        show_func = show_n(int(show_str), reverse=show_top)
    
    return show_func

if __name__ == '__main__':
    main()