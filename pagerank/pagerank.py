import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create empty dictionary to add page:probability pairs
    key = {}
    # Number of links on the page and in the corpus
    NumLinksi = len(corpus[page])
    N = len(corpus)
    # Iterate through the corpus and assign a probability value to each page
    for p in corpus:
        if NumLinksi == 0:
            PRp = 1/N
        elif p in corpus[page]:
            PRp = (1-damping_factor)/N + damping_factor/NumLinksi
        else:
            PRp = (1-damping_factor)/N
        key.update({p:PRp})
    return key

    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Choose a page at random from the corpus
    page = random.choice(list(corpus.keys()))
    # Create an empty counter for the pages
    page_counter = {}
    for page_i in list(corpus.keys()):
        page_counter.update({page_i:0})
    # Iterate through n number of trials, updating count for the 
    # current page and randomly choosing a new page using weights from
    # the transition model
    for _ in range(n):
        page_counter[page] += 1/n
        weights = list(transition_model(corpus,page,damping_factor).values())
        page = random.choices(list(corpus.keys()),weights=weights)[0]
    return page_counter



    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize pages with probability of 1/N
    prob = {}
    change = {}
    # Establish common variables
    for page in list(corpus.keys()):
        prob.update({page:(1/len(corpus))})
        change.update({page:999})
    # Begin iterative algorithm
    while max(list(change.values())) > 0.001:
        for page in list(corpus.keys()):
            summation = 0
            for page_iter in list(corpus.keys()):
                # Case if no links on page, treat as equal probability to link to any page, including itself
                if len(corpus[page_iter]) == 0:
                    summation += 1/len(corpus)
                # Case if a page links back to the current page
                elif page in corpus[page_iter]:
                    summation += prob[page_iter]/len(corpus[page_iter])
            temp_prob = (1-damping_factor)/len(corpus) + damping_factor*summation
            change[page] = abs(prob[page]-temp_prob)
            prob[page] = temp_prob
    return prob
            
            
    raise NotImplementedError


if __name__ == "__main__":
    main()
