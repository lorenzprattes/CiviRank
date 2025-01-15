from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from typing import Dict, Any
from civirank import analyzers, parsers, rankers
from ranking_challenge.request import RankingRequest
import argparse
import uvicorn
import json
from datetime import datetime

class Comment(BaseModel):
    id: str = Field(
        description="A unique ID describing a specific piece of content. We will do our best to make an ID for a given item persist between requests, but that property is not guaranteed."
    )
    parent_id: Optional[str] = Field(
        description="For threaded comments, this identifies the comment to which this one is a reply. Blank for top-level comments.",
        default=None,
    )
    title: Optional[str] = Field(
        description="The post title, only available on reddit posts.", default=None
    )
    text: str = Field(
        description="The text of the content item. Assume UTF-8, and that leading and trailing whitespace have been trimmed."
    )

class Comments(BaseModel):
    comments: list[Comment] = Field(description="The content items to be ranked.")

class RankingResponse(BaseModel):
    """A response to a ranking request"""

    ranked_ids: Dict[str, int] = Field(
        description="A dictionary where the keys are the IDs of the content items and the values are their ranks."
    )
    warning_index: Optional[int] = Field(
        description="The index of the first item that should trigger a warning to the user. If no warning is needed, this field should be omitted.",
        default=None
    )

app = FastAPI(
    title="Ranking Challenge",
    description="Ranks input using a local ranker.",
    version="0.1",
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking Challenge')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--scroll_warning_limit', type=float, default=-0.1, help='Scroll warning limit')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--no-download-models', action='store_false', dest='download_models', help='Skip downloading models, use models from disk instead')

    args = parser.parse_args()

    # Initialize the ranker instance with the download_models argument
    ranker = rankers.LocalRanker(download_models=args.download_models)

    # Define routes after initializing the ranker
    @app.post('/rank')
    def rank(ranking_request: RankingRequest) -> RankingResponse:
        print("Hello? Anyone there?", flush=True)
        print("Ranking request recieved")

        for item in ranking_request.items:
            print(item.id, flush=True)

        ranked_results, new_items = ranker.rank(ranking_request, batch_size=args.batch_size, scroll_warning_limit=args.scroll_warning_limit)
        return {"ranked_ids": ranked_results, "new_items": new_items}

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}
    
    @app.post('/rank_comments')
    def rank(request: Comments) -> RankingResponse:
        print("Comment ranking request recieved", flush=True)
        for item in request.comments:
            print(item.id, flush=True)
        ranked_results, insert_index = ranker.rank_comments(request.comments, batch_size=args.batch_size, scroll_warning_limit=args.scroll_warning_limit)
        return {"ranked_ids": ranked_results, "warning_index": insert_index}




    uvicorn.run(app, host='0.0.0.0', port=args.port, log_level='warning')
