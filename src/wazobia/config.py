from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field

class Settings(BaseSettings):
    hf_token: str
    hf_dataset_repo: str = Field(
        validation_alias=AliasChoices("HF_DATASET_REPO", "HF_BUCKET")
    )
    model_config = SettingsConfigDict(env_file='.env')


settings = Settings() # type: ignore
