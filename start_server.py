from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from civirank import analyzers, parsers, rankers
from ranking_challenge.request import RankingRequest
import argparse
import uvicorn
import json
from datetime import datetime
from utils import Comment, Comments, RankingResponse



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
    parser.add_argument('--language', type=str, default='en', help='Language for the models')

    args = parser.parse_args()

    # Initialize the ranker instance with the download_models argument
    ranker = rankers.LocalRanker(language=args.language)

    # Define routes after initializing the ranker
    @app.post('/rank_comments')
    def rank(request: Comments) -> RankingResponse:
        print("Comment ranking request recieved", flush=True)
        #todo remove when finished, useful for comparison for now #loveprintfdebugging
        for item in request.comments:
            print(item.id, flush=True)
        ranked_results, insert_index = ranker.rank_comments(request.comments, batch_size=args.batch_size, scroll_warning_limit=args.scroll_warning_limit)
        return {"ranked_ids": ranked_results, "warning_index": insert_index}

    uvicorn.run(app, host='0.0.0.0', port=args.port, log_level='warning')
