import wiki_vector_search as wvs


def test_hello():
    assert wvs.WVS().greet("user") == "Hello, User!"
