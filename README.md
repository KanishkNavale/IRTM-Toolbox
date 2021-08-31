# Information Retrieval & Text Mining Toolbox

This repository holds functions pivotal for IRTM processing. This repo. is staged for continuous development.

## Quick Install using 'pip/pip3' & GitHub

```bash
pip install git+https://github.com/KanishkNavale/IRTM-Toolbox.git
```

## Import Module

```python
from irtm.toolbox import *
```

## Using Functions

1. Soundex: A phonetic algorithm for indexing names by sound, as pronounced in English.

    ```python
    print(soundex('Muller'))
    print(soundex('Mueller'))
    ```

    ```bash
    >>> 'M466'
    >>> 'M466'
    ```

2. Tokenizer: Converts a sequence of characters into a sequence of tokens.

    ```python
    print(tokenize('LINUX'))
    print(tokenize('Text Mining 2021'))
    ```

    ```bash
    >>> ['linux']
    >>> ['text', 'mining']
    ```

3. Vectorize: Converts a string to token based weight tensor.

    ```python
    vector = vectorize([
            'texts ([string]): a multiline or a single line string.',
            'dict ([list], optional): list of tokens. Defaults to None.',
            'enable_Idf (bool, optional): use IDF or not. Defaults to True.',
            'normalize (str, optional): normalization of vector. Defaults to l2.',
            'max_dim ([int], optional): dimension of vector. Defaults to None.',
            'smooth (bool, optional): restricts value >0. Defaults to True.',
            'weightedTf (bool, optional): Tf = 1+log(Tf). Defaults to True.',
            'return_features (bool, optional): feature vector. Defaults to False.'
            ])

    print(f'Vector Shape={vector.shape}')
    ```

    ```bash
    >>> Vector Shape=(8, 37)
    ```

4. Predict Token Weights: Computes importance of a token based on classification optimization.

    ```python
    dictionary = ['vector', 'string', 'bool']
    vector = vectorize([
            'X ([np.array]): vectorized matrix columns arraged as per the dictionary.',
            'y ([labels]): True classification labels.',
            'epochs ([int]): Optimization epochs.',
            'verbose (bool, optional): Enable verbose outputs. Defaults to False.',
            'dict ([type], optional): list of tokens. Defaults to None.'
            ], dict=dictionary)

    labels = np.random.randint(1, size=(vector.shape[0], 1))
    weights = predict_weights(vector, labels, 100, dict=dictionary)
    ```

    ```bash
    >>> Token-Weights Mappings: {'vector': 0.22097790924850977, 
                                 'string': 0.39296369957440075, 
                                 'bool': 0.689853175081446}
    ```
