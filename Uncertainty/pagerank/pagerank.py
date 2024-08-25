import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    # corpus = crawl(sys.argv[1])
    corpus = {'1': {'2'}, '2': {'1', '3'}, '3': {'2', '5', '4'}, '4': {'1', '2'}, '5': set()}
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

    num_pages = len(corpus)

    if not corpus[page]:
        return {page: 1 / num_pages for page in corpus}

    distribution = {page: (1 - damping_factor) / num_pages for page in corpus}
    linked_pages = corpus[page]
    num_links = len(linked_pages)

    for linked_page in linked_pages:
        distribution[linked_page] += damping_factor / num_links

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page_rank = {page: 0.0 for page in corpus.keys()}
    current_page = random.choice(list(corpus.keys()))
    page_rank[current_page] += 1

    for i in range(n):
        transition_probs = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(transition_probs.keys()), list(transition_probs.values()), k=1)[0]
        page_rank[current_page] += 1

    total_samples = sum(page_rank.values())
    page_rank = {page: rank / total_samples for page, rank in page_rank.items()}

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_ranks = {page: 1 / num_pages for page in corpus.keys()}

    while True:
        previous_ranks = page_ranks.copy()
        count = 0
        for page in page_ranks:
            new_rank = calculate_page_rank(page, damping_factor, corpus, page_ranks)
            if abs(new_rank - previous_ranks[page]) < 0.001:
                count += 1
            page_ranks[page] = new_rank
        if count == num_pages:
            break

    return prob_distribution(page_ranks)


def prob_distribution(page_ranks):
    total = sum(page_ranks.values())
    for page, rank in page_ranks.items():
        page_ranks[page] = rank/total
    return page_ranks


def calculate_page_rank(page, damping_factor, corpus, page_ranks):
    num_pages = len(corpus)
    incoming_links = [p for p, links in corpus.items() if page in links]

    rank_sum = sum(page_ranks[incoming_page] / num_links(corpus, incoming_page) for incoming_page in incoming_links)
    return (1-damping_factor) / num_pages + damping_factor * rank_sum


def num_links(corpus, page):
    return len(corpus[page]) if corpus[page] else len(corpus)


if __name__ == "__main__":
    main()
