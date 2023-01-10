import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Iterate through directory and use os library to join filenames and convert to string
    txtdict = dict()
    for txt in os.listdir(directory):
        with open(os.path.join(directory,txt)) as txtfile:
            page = txtfile.read()
            txtdict.update({txt:page})
    return txtdict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.translate(str.maketrans("","",string.punctuation))
    txtlist = list(nltk.tokenize.word_tokenize(document))
    txtlist = list(map(str.lower,txtlist))
    stopwords = nltk.corpus.stopwords.words('english')
    filtered = [word for word in txtlist if word not in stopwords]
    return filtered


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Empty set of words and ditionary for idf values
    words = set()
    idf = dict()

    # Iterate and add all words to set
    for file in documents:
        words = words.union(set(documents[file]))

    # Iterate and calculate idf values
    for word in words:
        totdocs = len(documents) # Total documents
        numcont = 0 # Number of documents that have the word
        for doc in documents:
            if word in documents[doc]:
                numcont += 1
        idfcalc = math.log(totdocs/numcont)
        idf.update({word:idfcalc})
        
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Empty list for tfidfs values
    tfidfs = []

    # Assuming query is a set() - compute tfidfs values of each file
    for file in files:
        temp_tfidfs = 0
        for word in query:
            if word in idfs.keys():
                temp_tfidfs += (files[file].count(word) * idfs[word])
        tfidfs.append((file,temp_tfidfs))
    
    # Sort tfidfs values
    tfidfs.sort(key=lambda tfidf:tfidf[1],reverse=True)

    # Extract file names only
    topfiles = [tfidf[0] for tfidf in tfidfs]

    return topfiles[0:n] 


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Empty list for idf values
    idf = []

    # Assuming query is a set() - compute idf values of each sentence
    for sentence in sentences:
        temp_idf = 0
        num_terms = 0
        for word in query:
            if word in sentences[sentence]:
                temp_idf += idfs[word]
                num_terms += sentences[sentence].count(word)
        term_density = num_terms / len(sentences[sentence])
        idf.append((sentence,temp_idf,term_density))
    
    # Sort idf values by idf value first and then term density
    idf.sort(key=lambda sorter:(sorter[1],sorter[2]),reverse=True)

    # Extract file names only
    topsentences = [sentence[0] for sentence in idf]
    return topsentences[0:n] 



if __name__ == "__main__":
    main()
