# Information Retrieval & Text Mining Toolbox

This repository holds functions pivotal for IRTM processing. This repo. is staged for continuous development.


## Import Module

```python
from IRTM import *
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

2. Tokenizer: Convert a sequence of characters into a sequence of tokens.

    ```python
    print(tokenize('LINUX'))
    print(tokenize('Text Mining 2021'))
    ```

    ```bash
    >>> ['linux']
    >>> ['text', 'mining']
    ```
