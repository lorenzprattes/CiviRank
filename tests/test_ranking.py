import pytest
import sys
sys.path.append('../civirank')
from civirank.rankers import LocalRanker
import pandas as pd
import os
import json

@pytest.fixture
def input_files():
    # Path to the directory containing test input files
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    input_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.json')]
    return input_files

@pytest.fixture
def ranker():
    # Initialize the LocalRanker
    return LocalRanker(download_models=False)

def test_rank_comments(ranker, input_files):
    for file_path in input_files:
        with open(file_path, 'r') as f:
            comments_data = json.load(f)


        # Call the rank_comments method
        ranked_results = ranker.rank(comments_data)

        # Check if the output is not empty
        assert ranked_results is not None
        #assert len(ranked_results) == len(comments_data)

        # Optionally, you can add more assertions to check the content of the results
        #for result in ranked_results:
        #    assert "compound_score" in result

# Run the tests
if __name__ == "__main__":
    pytest.main()