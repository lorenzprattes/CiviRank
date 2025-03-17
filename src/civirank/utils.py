from pydantic import BaseModel, Field
from typing import Optional
from typing import Dict, Any

class Comment(BaseModel):
    id: str = Field(
        description="A unique ID describing a specific piece of content."
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
    warning_index: int = Field(
        description="The index of the first item that should trigger a warning to the user. If no warning is needed, this field is set to -1",
        default=None
    )