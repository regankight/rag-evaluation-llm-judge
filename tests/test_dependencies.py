"""Guards for dependency problems found in Phase 0 (docs/decisions/0002)."""


def test_ragas_imports_with_pinned_langchain_community():
    # langchain-community 0.4.2 removed a module RAGAS 0.4.3 imports; the pin to 0.4.1
    # keeps `import ragas` working. If this fails, the pin was lost or loosened.
    import ragas
    from ragas.metrics.collections import Faithfulness  # noqa: F401

    assert ragas.__version__ == "0.4.3"
