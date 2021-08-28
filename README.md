# Information Retrieval & Text Mining Toolbox

This repository holds functions pivotal for IRTM processing. This repo. is staged for continuous development.

## Functions

1. Soundex: A phonetic algorithm for indexing names by sound, as pronounced in English.

    ```python
    from IRTM import soundex

    print(soundex('Muller'))
    print(soundex('Mueller'))
    ```
    ```
    >>> 'M466'
    >>> 'M466'
    ```