import json
from typing import Any

from pathlib import Path
from wazobia.ledger import entity

class JSONRepo:
    def __init__(self, root_dir : str, repo: entity.Repo, strict=False):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        fp =  f"{repo.repo_id}_{repo.repo_type}.json"
        self.path = self.root_dir / fp
        if not self.path.exists():
            if strict:
                raise ValueError(f"{str(self.path)} does not exist")
            self._create_json(self.path, repo)
        self.repo = repo

    def _create_json(self, path: Path, repo: entity.Repo):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(repo.model_dump(), file)

    def _load_json(self, path: Path, as_dict: bool = False) -> Any:
        with path.open("r", encoding="utf-8") as file:
            if as_dict:
                return dict(json.load(file))
            return entity.Repo(**json.load(file))

    def _save_json(self,path : Path, repo: entity.Repo):
        repo = entity.Repo.model_validate(repo)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(repo.model_dump(), file)
            file.flush()
            import os
            os.fsync(file.fileno())
        tmp.replace(path)

    def add(self, entity_state: entity.Entity, exist_ok: bool = True):
        repo = self._load_json(self.path)
        exists = any(self._filter_entity(repo.model_dump(), entity_state.filename, by="filename"))
        if exists:
            if not exist_ok:
                raise ValueError(f"Entity already exists: {entity_state.filename}")
            return
        repo.entities.append(entity_state)
        self._save_json(self.path, repo)
        

    def _filter_entity_fn(self, entity_obj : dict, value: str, key: str):
        return entity_obj[key] == value

    def _filter_entity(self, repo_obj: dict[str, Any], value: str, by="status", return_index=False):
        if return_index:
            return filter(
                lambda item: self._filter_entity_fn(item[1], value=value, key=by),
                enumerate(repo_obj["entities"]),
            )
        else:
            return filter(
                lambda item: self._filter_entity_fn(item, value=value, key=by),
                repo_obj["entities"],
            )

    def filter_entity(self, value: str, by="status", return_index=False):
        repo = self._load_json(self.path, as_dict=True)
        return list(self._filter_entity(repo, value, by, return_index))

    def _update_entity(self,repo, fp, entries: dict):
        try:
            ind, _ = next(self._filter_entity(repo, by="filename",value=fp, return_index=True))
        except StopIteration as exc:
            raise KeyError(f"Ledger entity not found: {fp}") from exc
        for key, value in entries.items():
            repo["entities"][ind][key] = value
        self._save_json(self.path, repo)

    def update_entity(self, fp, entries: dict):
        repo = self._load_json(self.path, as_dict=True)
        self._update_entity(repo, fp, entries)
