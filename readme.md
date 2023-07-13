## RepoSearch
### Description

RepoSearch is a tool for searching through repositories of text and source code using natural language queries, based on embeddings from a custom-specified model.

Current options for model are:
* [Instructor](https://huggingface.co/hkunlp/instructor-large) for local generation (default, GPU recommended)
* [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) for remote generation

### Install

#### Requirements
Pip should install missing requirements automatically. Though you may want to install the following ahead of time to speed up the process:
* [PyTorch](https://pytorch.org/)
* [sentence-transformers](https://pypi.org/project/sentence-transformers/)
* [HuggingFace Datasets](https://huggingface.co/docs/datasets/installation)

#### Install Steps
1. Open a terminal.
2. *[Optional]* Create a virtual/conda/whatever environment.
3. `pip install git+https://github.com/Netruk44/repo-search`
4. `repo_search --help`

### Arguments

`repo_search <generate|query> <repository_name> <arguments>`

| Argument | Description |
| -------- | ----------- |
| `generate` | Generate embeddings for a repository. |
| `query` | Query a repository for files similar to the given query. |
| `repository_name` | The name for a collection of embeddings |

---

#### Optional Shared Arguments

| Argument | Description |
| -------- | ----------- |
| `--model_type` | The type of model to use for generating or querying embeddings. See [Available Model Types](#available-model-types) for more information.<br /><br />**Default**: `instructor` |
| `--model_name` | The name of the model to use for generating or querying embeddings. Options available depend on model type. |
| `--embeddings_dir` | The directory to store the generated embeddings in. <br /><br />**Default**: An `embeddings` directory located in the folder RepoSearch was installed to |
| `--verbose` | Whether or not to print verbose output.<br /><br />**Default**: Off |

---

#### Generate Arguments
`repo_search generate <repository_name> <repository_source>`

`repository_source` can be one of:
* A path to a local directory
* A path to a local zip file
* A URL to a zip file to download
* A URL to a GitHub repository to download (the `main` branch is downloaded)

---

#### Query Arguments
`repo_search query <repository_name> <query>`

`query` is a string containing the query to search for.

### How does it work?
For each file in the repository, the embeddings are sent to a customizable model (default: [instructor-large](https://huggingface.co/hkunlp/instructor-large)) to generate an embedding. If a file is too long to fit within a single embedding, it is split into smaller chunks and each chunk is embedded separately.

The retrieved embeddings are stored in a [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) dataset. Check out [the schema](#dataset-schema) for more information about using the generated dataset.

> **Possible TODO**: *The embeddings are indexed using [FAISS](https://faiss.ai/), which allows for fast nearest neighbor searches to your queries.*

### Example Usage

#### Local Repository
*Generating embeddings from a local copy of the OpenMW (open source game engine) repository, then querying it.*

```bash
$ repo_search generate openmw ~/Developer/openmw/apps --verbose
Loading libraries...
Generating embeddings from local directory /Users/danielperry/Developer/openmw/apps for openmw...
WARNING: Could not read as text file: /Users/danielperry/Developer/openmw/apps/openmw_test_suite/toutf8/data/french-win1252.txt
WARNING: Could not read as text file: /Users/danielperry/Developer/openmw/apps/openmw_test_suite/toutf8/data/russian-win1251.txt
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1386/1386 [05:53<00:00,  3.92it/s]

$ repo_search query openmw "NPC navigation code and examples on making an NPC navigate towards a specific destination."
Loading libraries...
Querying embeddings...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1386/1386 [00:00<00:00, 3375.94it/s]
0.7708185244777552: openmw/mwmechanics/aiwander.cpp
0.7648722322559: openmw/mwmechanics/obstacle.cpp
0.7593991785701977: openmw/mwmechanics/aipursue.cpp
0.7570192805465497: openmw/mwmechanics/aiescort.cpp
0.7540042527421098: openmw/mwmechanics/aiescort.hpp
0.7534971127334509: openmw/mwmechanics/aitravel.cpp
0.7531876754013663: openmw/mwmechanics/aiface.cpp
0.7529779124249095: openmw/mwmechanics/aipackage.cpp
0.7527566874958713: openmw/mwworld/actionteleport.cpp
0.7491856749386254: openmw/mwmechanics/pathfinding.cpp
```

#### Zip File Download

*Downloading the latest state of the Borg Backup repository from GitHub, generating embeddings, then querying it.*

```bash
$ repo_search generate borg https://github.com/borgbackup/borg/archive/refs/heads/master.zip --verbose
Loading libraries...
Downloading https://github.com/borgbackup/borg/archive/refs/heads/master.zip...
Generating embeddings from zipfile for borg...
WARNING: Could not read as text file: borg-master/docs/_static/favicon.ico
WARNING: Could not read as text file: borg-master/docs/_static/logo.pdf
WARNING: Could not read as text file: borg-master/docs/_static/logo.png
WARNING: Could not read as text file: borg-master/docs/internals/compaction.odg
WARNING: Could not read as text file: borg-master/docs/internals/compaction.png
...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 425/425 [02:17<00:00,  3.09it/s]

$ repo_search query borg "Code implementing file chunking and deduplication."
Loading libraries...
Querying embeddings...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 425/425 [00:00<00:00, 3524.80it/s]
0.7795432405745771: borg-master/scripts/fuzz-cache-sync/testcase_dir/test_simple
0.776652925136563: borg-master/src/borg/chunker.pyx
0.7625068003811171: borg-master/docs/usage/notes.rst
0.7601486530920606: borg-master/docs/misc/internals-picture.txt
0.7592314451132307: borg-master/src/borg/hashindex.pyi
0.7587992123424291: borg-master/src/borg/chunker.pyi
0.7545961178472722: borg-master/src/borg/testsuite/chunker.py
0.748955970602009: borg-master/src/borg/_chunker.c
0.7468354876619363: borg-master/src/borg/cache.py
0.7418049390391546: borg-master/src/borg/_hashindex.c
```

### Dataset Schema
The generated dataset consists of just two columns.

> | Column Name | Description |
> | ----------- | ----------- |
> | `file_path` | The path to the file that was embedded. Useful for displaying to the user. |
> | `embeddings` | An array of embeddings for the file. An empty array indicates an error occurred when generating embeddings for the file. The array may have one or more embeddings within it, depending on the source file length. |

### Available Model Types

`--model_type` specifies which model should be used to generate the embeddings. Currently there are two options: `instructor` and `openai`.

#### Instructor (Default)

> `--model_type instructor`
> 
> By default, RepoSearch uses [instructor-large](https://huggingface.co/hkunlp/instructor-large) to generate the embeddings. 
> 
> **`--model_name`**:
> | Model Name | Description |
> | ------------ | ----------- |
> | `hkunlp/instructor-large` | The default model. |
> | `hkunlp/instructor-xl` | A larger version of the default model. |

---

#### OpenAI

> `--model_type openai`
> 
> **Note**: Using this model type requires you to supply your own OpenAI API key!
> 
> `export OPENAI_API_KEY=sk-...`
> 
>>  **Warning**: **DO NOT** use this project with any sensitive code/data! The contents of all files will be sent to OpenAI's API for embedding generation.
>
> **`--model_name`**: 
> | Model Name | Description |
> | ------------ | ----------- |
> | `text-embedding-ada-002` | The default model. |
> 
> **Cost**:
> Cost per query is negligible, almost always less than 1/10th of a penny unless you're writing paragraphs of text.
> 
> Generating embeddings:
> * For the [OpenMW](https://gitlab.com/OpenMW/openmw) repository (generating embeddings for ~9 MB worth of source files) costs ~$0.20 USD.
> * For the [Borg Backup](https://github.com/borgbackup/borg) repository (<5 MB of source) costs ~$0.10 USD.