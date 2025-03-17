import pytest
import sys
sys.path.append('../civirank')
from civirank.ranker import CiviRank
import pandas as pd
import os
import json
from pydantic.tools import parse_obj_as
sys.path.append('..')
from civirank.utils import Comments, Comment
from civirank import CiviRank


@pytest.fixture
def input_files_platforms():
    # Path to the directory containing test input files
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    input_files_platforms = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.json') and not f.startswith('remark')]
    return input_files_platforms

@pytest.fixture
def input_comments():
    # Path to the directory containing test input files
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    input_comments = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.json') and f.startswith('remark')]
    return input_comments

@pytest.fixture
def ranker():
    # Initialize the LocalRanker
    return CiviRank()

def test_rank(ranker, input_files_platforms):
    for file_path in input_files_platforms:
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

def test_rank_comments(ranker, input_comments):
    for file_path in input_comments:
        with open(file_path, 'r') as f:
            comments_data = json.load(f)
            #comments = Comments.model_validate_json(comments=comments_data)
        comments = parse_obj_as(Comments, comments_data)

        # Call the rank_comments method
        ranked_results = ranker.rank_comments(comments.comments)

        # Check if the output is not empty
        assert ranked_results is not None
        #assert len(ranked_results) == len(comments_data)

        comments_map = {comment.id: comment for comment in comments.comments}

        # Reorder comments_data based on the order of IDs in ranked_results
        ordered_comments = [comments_map[ranked.id] for ranked in ranked_results]
        print(ordered_comments)
        # Optionally, you can add more assertions to check the content of the results
        #for result in ranked_results:
        #    assert "compound_score" in result
# Run the tests
if __name__ == "__main__":
    pytest.main()