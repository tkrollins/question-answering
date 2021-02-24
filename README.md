Context-Finder
==============================

Context-Finder is a proposed model which takes a text document of variable sized contexts (paragraphs) and scores each context with the probability in which it will contain the answer to a given question.  It then aggregates these scores to find the most likely answer-containing context given the question. The Stanford Question Answering Dataset (SQuAD) was used for training and evaluation of the model.  As opposed to most models using SQuAD, rather than predicting an answer-containing span given a context, Context-Finder predicts an answer-containing context given a document.  As input features, the model utilizes Sentence-BERT sentence embeddings as well as a score for both named entities and verb roots of both the question and context text.  The model is comprised of dot product attention mechanisms as well as a feed-forward neural network.  Its performance was subsequently compared against the DrQA Document Reader, an established model with a similar objective, after both were trained using the same data.  Context-Finder achieved an exact context prediction accuracy within 1% of the DrQA Document reader.  Context-Finder achieves this while eliminating the use of recurrent neural networks and token-level embeddings, which can significantly improve performance.  Additionally, the correct context was found within the top five scoring candidates of document more than 83% of the time.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
