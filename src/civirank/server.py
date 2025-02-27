from .ranker import CiviRank
from .utils import Comments, RankingResponse
from fastapi import FastAPI
import argparse
import uvicorn


def create_server() -> FastAPI:
  if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking Challenge')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--scroll_warning_limit', type=float, default=-0.1, help='Scroll warning limit')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--language', type=str, default='en', help='Language for the models')
    parser.add_argument('--model_id', type=str, help='Model ID for the toxicity analyzer')


  available_models = ["celadon", "jagoldz/gahd", "textdetox/xlmr-large-toxicity-classifier"]
  available_languages = ["ger", "en"]
  args = parser.parse_args()

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

  @app.post('/rank_comments')
  def rank(request: Comments) -> RankingResponse:
    print("Comment ranking request recieved", flush=True)
    #todo remove when finished, useful for comparison for now #loveprintfdebugging
    for item in request.comments:
        print(item.id, flush=True)
    ranked_results, insert_index = ranker.rank(request.comments, batch_size=args.batch_size, scroll_warning_limit=args.scroll_warning_limit)
    return {"ranked_ids": ranked_results, "warning_index": insert_index}

  return app

def run_server():
  app = create_server()
  uvicorn.run(app, host='0.0.0.0', port=app.state.port, log_level='warning')


if __name__ == "__main__":
  run_server()
