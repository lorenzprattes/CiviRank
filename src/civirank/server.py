from .ranker import CiviRank
from .utils import Comments, RankingResponse
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
import argparse
import uvicorn
from threading import Lock


def create_server() -> FastAPI:
  if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking Challenge')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--scroll_warning_limit', type=str, help='Scroll warning limit')
    parser.add_argument('--language', type=str, default='en', help='Language for the models')
    parser.add_argument('--model_id', type=str, help='Model ID for the toxicity analyzer')


  available_models = ["celadon", "jagoldz/gahd", "textdetox/xlmr-large-toxicity-classifier"]
  available_languages = ["ger", "en"]
  args = parser.parse_args()

  #try parsing scroll_warning_limit into float
  if args.scroll_warning_limit != None: 
    try:
      args.scroll_warning_limit = float(args.scroll_warning_limit)
    except ValueError:
      args.scroll_warning_limit = None

  if args.language == "" or args.language == None:
      args.language = "en"
  if args.model_id == "" or args.model_id == None:
      args.model_id = "celadon"
  elif args.model_id not in available_models:
      raise ValueError(f"Model ID {args.model_id} not in available models {available_models}")

  ranker = CiviRank(language=args.language, model_id=args.model_id)

  app = FastAPI(
      title="Ranking Challenge",
      description="Ranks input using a local ranker.",
      version="0.1",
  )

  app.state.ranker = ranker
  app.state.port = args.port
  app.state.scroll_warning_limit = args.scroll_warning_limit
  lock = Lock()
  app.state.lock = lock

  @app.post('/rank_comments')
  def rank(request: Comments) -> RankingResponse:
    print("Comment ranking request recieved", flush=True)
    #todo remove when finished, useful for comparison for now #loveprintfdebugging
    for item in request.comments:
      print(item.id, flush=True)
    with app.state.lock:
      ranked_results, insert_index = ranker.rank(request.comments, scroll_warning_limit=app.state.scroll_warning_limit)
    print("Ranking finished", flush=True)
    print("Warning index:", insert_index, flush=True)
    return {"ranked_ids": ranked_results, "warning_index": insert_index}

  return app

def run_server():
  app = create_server()
  uvicorn.run(app, host='0.0.0.0', port=app.state.port, log_level='warning')


if __name__ == "__main__":
  run_server()
