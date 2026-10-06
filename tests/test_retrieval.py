from retrieval import BM25Index
from tydiqa import Passage


def p(i, title, text):
    return Passage(f"d:{i}", "d", title, f"https://x/{i}", i, 0, len(text), text)


INDEX = BM25Index(
    [
        p(0, "Japan", "The capital of Japan is Tokyo."),
        p(1, "France", "Paris is the capital of France."),
        p(2, "Cheese", "Cheddar cheese comes from Somerset."),
    ]
)


def test_best_match_first_with_provenance():
    hits = INDEX.search("capital of Japan", k=2)
    assert hits[0].passage.passage_id == "d:0"
    assert hits[0].passage.url == "https://x/0" and hits[0].rank == 1
    assert hits[0].score >= hits[-1].score


def test_title_is_searchable():
    assert INDEX.search("cheese", k=1)[0].passage.title == "Cheese"


def test_no_matching_words_returns_nothing_not_arbitrary_passages():
    assert INDEX.search("zzzz qqqq", k=3) == []


def test_only_matching_passages_returned_even_if_k_is_larger():
    hits = INDEX.search("Somerset", k=3)
    assert [h.passage.passage_id for h in hits] == ["d:2"]


def test_k_larger_than_corpus_is_safe():
    assert len(INDEX.search("capital", k=50)) == 2
