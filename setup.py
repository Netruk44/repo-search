from setuptools import setup, find_packages
exec(open('repo_search/version.py').read())

setup(
    name='repo-search',
    packages = find_packages(exclude=[]),
    include_package_data = True,
    entry_points = {
        'console_scripts': [
            'repo_search = repo_search.cli:main'
        ],
    },
    version = __version__,
    description = 'Search through code repositories with natural language queries.',
    author = 'Daniel Perry',
    author_email = 'python@danieltperry.me',
    url = 'https://github.com/netruk44/repo-search',
    install_requires = [
        'openai',
        'click',
        'datasets',
        'faiss-cpu',
        'tqdm',
    ]
)