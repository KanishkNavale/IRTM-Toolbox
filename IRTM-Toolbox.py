# Generate Soundex codes for token
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
            except Exception as e:
                print(f'Processing Error: {e}')

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
