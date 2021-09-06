###############################################################################
# LIBRARY IMPORTS
###############################################################################
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import deque


###############################################################################
# SOUNDEX
###############################################################################
def soundex(word):
    """
    Description,
        Soundex encode a token.

    Args:
        word (string): string for conversion.
    """
    if word.isalpha():

        # Clip the first value
        word = word.upper()
        head = word[0]
        word = word[1:].lower()

        # 1st rule imposition
        for i in ['a', 'e', 'i', 'o', 'u', 'w', 'y']:
            word = word.replace(i, '0')

        # 2nd rule imposition
        for i in ['b', 'f', 'p', 'v']:
            word = word.replace(i, '1')

        # 3rd rule imposition
        for i in ['c', 'g', 'j', 'k', 'q', 's', 'x', 'z']:
            word = word.replace(i, '2')

        # 4th rule imposition
        word = word.replace('l', '4')

        # 5th rule imposition
        for i in ['m', 'n']:
            word = word.replace(i, '5')

        # 6th rule imposition
        word = word.replace('r', '6')

        # Remove repeatative letters
        word = list(word)
        j = 0
        for i in range(len(word)):
            if (word[j] != word[i]):
                j += 1
                word[j] = word[i]

        # Remove all 0s
        for i in range(len(word)):
            try:
                word.remove('0')
            except:
                pass

        # Add the header in
        word.insert(0, head)

        # Padding length
        if len(word) > 4:
            soundex = ''.join(word[:4])
            return(soundex.upper())
        else:
            for i in range(4):
                if len(word) < 4:
                    word.insert(-1, '0')
                else:
                    break

            soundex = ''.join(word)
            return(soundex.upper())

    else:
        print('Input not valid for soundex processing!')


###############################################################################
# TOKENIZE
###############################################################################
def tokenize(word):
    """
    Description:
        Tokenizes a string.

    Args:
        word ([str]): str

    Returns:
        [list]: list of tokens.
    """
    tokens = word_tokenize(word)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]
    return tokens


###############################################################################
# Vectorizer
###############################################################################
def vectorize(texts, dict=None, enable_Idf=True,
              normalize='l2', max_dim=None,
              smooth=True, weightedTf=True, return_features=False):
    """
    Description:
        Creates weights tensor based on parsed string.

    Args:
        texts ([string]): a multiline or a single line string
        dict ([list], optional): list of tokens. Defaults to None.
        enable_Idf (bool, optional): use IDF or not. Defaults to True.
        normalize (str, optional): normalization of vector. Defaults to 'l2'.
        max_dim ([int], optional): dimension of vector. Defaults to None.
        smooth (bool, optional): restricts value >0. Defaults to True.
        weightedTf (bool, optional): Tf = 1+log(Tf). Defaults to True.
        return_features (bool, optional): feature vector. Defaults to False.

    Returns:
        [np.matrix]: vectorized weight matrix
        [list]: feature vectors
    """
    if dict is None:
        vectorizer = TfidfVectorizer(use_idf=enable_Idf,
                                     norm=normalize, max_features=max_dim,
                                     sublinear_tf=weightedTf,
                                     smooth_idf=smooth)
    else:
        vectorizer = TfidfVectorizer(vocabulary=dict, use_idf=enable_Idf,
                                     norm=normalize, max_features=max_dim,
                                     sublinear_tf=weightedTf,
                                     smooth_idf=smooth)

    vector = vectorizer.fit_transform(texts)

    if return_features:
        return vector.todense(), vectorizer.get_feature_names()
    else:
        return vector.todense()


###############################################################################
# PREDICT WEIGHTS
###############################################################################
def predict_weights(X, y, epochs, verbose=False, dict=None):
    """
    Description:
        Predicts importance of a token based on classification optimization.

    Args:
        X ([np.array]): vectorized matrix columns arraged as per the dictionary.
        y ([labels]): True classification labels.
        epochs ([int]): Optimization epochs.
        verbose (bool, optional): Enable verbose outputs. Defaults to False.
        dict ([type], optional): list of tokens. Defaults to None.

    Returns:
        [dictionary]: Mappings of token & it's weights
    """

    W = np.random.uniform(0, 1, (1, X.shape[1]))
    v = np.zeros(W.shape)
    loss_log = deque(maxlen=3)

    for i in range(loss_log.maxlen):
        loss_log.append(0)

    for i in range(int(epochs)):
        pred_y = 1.0 / 1.0 + np.exp(X @ W.T)
        loss = -np.mean(np.log(pred_y.T) @ y)
        loss_log.append(loss)
        gradient = (pred_y - y).T @ X + (2.0 * 0.1 * W)
        v = (0.9 * v) + (1e-3 * gradient)
        W = W - v

        if verbose:
            if i % 100 == 0:
                print(f'Epoch={i} \t Loss={loss}')
            if np.mean(loss_log) == loss:
                print('Loss is not decreasing enough!')
                break
        else:
            if np.mean(loss_log) == loss:
                break

    mapping = {}
    weights = np.ravel(W)
    for i in range(len(dict)):
        mapping[dict[i]] = weights[i]

    return mapping


###############################################################################
# Page Rank
###############################################################################
def page_rank(tensor, teleportation=0.1, epochs=1000, return_TransMatrix=False):
    """
    Desciption,
        Computes Page Rank from Chain Matrix

    Args:
        tensor ([np.matrix]): Markov Chain Matrix
        teleportation (float, optional): Teleportation Rate. Defaults to 0.1.
        epochs (int, optional): Epochs of visits. Defaults to 1000.
        return_TeleMatrix (bool, optional): Returns Transition Matix. Defaults to False.

    Returns:
        rank: rank matrix
    """
    # Normalize the rows with N = no. of non-zero elements in row
    matrix = []
    for i in range(tensor.shape[0]):
        row = tensor[i].copy()
        N = np.count_nonzero(row)
        if N != 0:
            row = row / N
        matrix.append(row)
    tensor = np.vstack(matrix)

    # Divide the tensor by (1 - teleportation rate)
    tensor = (1 - teleportation) * tensor

    # Make the rows sum to '1'
    for i in range(tensor.shape[0]):
        row = tensor[i].copy()
        row += (1.0 - np.sum(row)) / row.shape[0]
        if np.sum(row) == 1.0:
            pass
        else:
            print(row)
        tensor[i] = row

    TransMatrix = tensor.copy()

    # Find Eigen Vector of the Matrix
    rank = (1 / tensor.shape[0]) * np.ones(tensor.shape[0])
    prev_rank = np.zeros(tensor.shape[0])

    for i in range(epochs):
        rank = rank @ np.power(TransMatrix, i+1)

        if np.array_equal(rank, prev_rank) or \
           np.linalg.norm(rank - prev_rank) <= 1e-6:
            break
        else:
            prev_rank = rank.copy()

    rank /= np.linalg.norm(rank)

    if return_TransMatrix:
        return np.around(rank, 4), TransMatrix
    else:
        return np.around(rank, 4)