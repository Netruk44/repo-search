## RepoSearch
### Description

RepoSearch is a tool for searching through repositories of text and source code using natural language queries, based on embeddings from a custom-specified model.

Current options for model are:
* [Instructor](https://huggingface.co/hkunlp/instructor-large) for local generation (default, GPU recommended but not required)
* [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) for remote generation

### Install

#### Install Steps
1. Open a terminal.
2. *[Optional]* Create a virtual/conda/whatever environment.
3. *[Optional]* Install [requirements](#requirements) into your environment.
3. `pip install git+https://github.com/Netruk44/repo-search`
4. `repo_search --help`

#### Requirements
Pip should install missing requirements automatically. Though you may want to install the following ahead of time to speed up the process:
* [PyTorch](https://pytorch.org/)
* [sentence-transformers](https://pypi.org/project/sentence-transformers/)
* [HuggingFace Datasets](https://huggingface.co/docs/datasets/installation)

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
$ repo_search generate openmw ~/Developer/openmw/apps
  <output trimmed for brevity>
100%|██████████████████████████████| 1386/1386 [05:53<00:00,  3.92it/s]

$ repo_search query openmw "Example of making an NPC navigate towards a specific destination."
  <output trimmed for brevity>
100%|██████████████████████████████| 1386/1386 [00:00<00:00, 3533.53it/s]

"Example of making an NPC navigate towards a specific destination."

89.55% match    openmw/mwmechanics/aipackage.cpp [25%-31% of the way through]
88.99% match    openmw/mwmechanics/aiwander.cpp [74%-78% of the way through]
87.65% match    openmw/mwmechanics/aipursue.cpp [33%-67% of the way through]
87.42% match    openmw/mwmechanics/aicombat.cpp [33%-38% of the way through]
87.19% match    openmw/mwmechanics/aitravel.cpp [40%-60% of the way through]
87.11% match    openmw/mwmechanics/pathfinding.cpp [82%-88% of the way through]
86.88% match    openmw/mwgui/dialogue.cpp [64%-68% of the way through]
86.81% match    openmw/mwmechanics/aipackage.hpp [40%-60% of the way through]
86.63% match    openmw/mwmechanics/aiwander.hpp [60%-80% of the way through]
86.30% match    openmw/mwmechanics/character.cpp [65%-66% of the way through]
```

#### Zip File Download + Embedding with OpenAI

*Downloading the latest state of the Borg Backup repository from GitHub, generating embeddings using OpenAI Embeddings, then querying it.*

```bash
$ repo_search generate borg https://github.com/borgbackup/borg/archive/refs/heads/master.zip --model_type openai
  <output trimmed for brevity>
100%|██████████████████████████████| 425/425 [02:17<00:00,  3.09it/s]

$ repo_search query borg "Code implementing file chunking and deduplication." --model_type openai
  <output trimmed for brevity>
100%|██████████████████████████████| 425/425 [00:00<00:00, 3524.80it/s]

"Code implementing file chunking and deduplication."

77.95% match    borg-master/scripts/fuzz-cache-sync/testcase_dir/test_simple [0%-100% of the way through]
77.67% match    borg-master/src/borg/chunker.pyx [0%-100% of the way through]
76.25% match    borg-master/docs/usage/notes.rst [0%-100% of the way through]
76.01% match    borg-master/docs/misc/internals-picture.txt [0%-100% of the way through]
75.92% match    borg-master/src/borg/hashindex.pyi [0%-100% of the way through]
75.88% match    borg-master/src/borg/chunker.pyi [0%-100% of the way through]
75.46% match    borg-master/src/borg/testsuite/chunker.py [0%-100% of the way through]
74.90% match    borg-master/src/borg/_chunker.c [0%-100% of the way through]
74.68% match    borg-master/src/borg/cache.py [50%-100% of the way through]
74.18% match    borg-master/src/borg/_hashindex.c [0%-100% of the way through]
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
> | `hkunlp/instructor-large` | The default model. Requires ~2.5 GB of VRAM to run. |
> | `hkunlp/instructor-xl` | A larger version of the default model. Requires ~6 GB of VRAM. |

---

#### OpenAI

> `--model_type openai`
> 
> **Note**: Using this model type requires you to supply your own OpenAI API key!
> 
> `export OPENAI_API_KEY=sk-...`
> 
>>  **Warning**: You should not use this model with any extremely sensitive code or data! The contents of all files will be sent to OpenAI's API for embedding generation.
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