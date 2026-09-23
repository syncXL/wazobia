from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    hf_token: str
    hf_bucket: str
    model_config = SettingsConfigDict(env_file='.env')


settings = Settings() # type: ignore
