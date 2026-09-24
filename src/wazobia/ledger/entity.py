from pydantic import BaseModel, Field

class Entity(BaseModel):
    filename : str = Field(description="Filepath on HF")
    status : str = Field(default="pending", description="Status")
    local_dir : list[str] = Field(default_factory=list, description="Local filename")

class Repo(BaseModel):
    repo_id : str = Field(description="Repo ID")
    repo_type : str = Field(description="Repo Type")
    entities : list[Entity] = Field(description="List of Files", default_factory=list)
