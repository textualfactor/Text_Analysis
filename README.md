### Textual Factor Analysis

This repository contains codes to implement the textual-factor framework developed in Cong, Liang and Zhang (2019). 

The repository is organized in the following way:

1. ``data`` folder contains the corpus of textual data that users wish to analyze. 

2. ``output`` folder contains the tokenized textual data that the code will generate. 

3. ``src`` folder contains the project's source codes of tokenization, clustering, and topic modeling. 

### Install

This project uses [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/), [FALCONN](https://github.com/falconn-lib/falconn/wiki) and [gensim](https://radimrehurek.com/gensim/). This version of the code also uses Google's pre-trained word2vec embeddings of words and phrases to generate clusters, so users should download them [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and place the file in the directory of this project. 

### Usage and Relevant Output

To run textual-factor analysis: 
```sh
$ python3 text_analysis.py
```

#### Update from previous version

* This code is abstracted version of the old code.

* The tokenization and clustering is multithreaded.
