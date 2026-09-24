from pathlib import Path
from wazobia.ledger import repo, entity

class CorpusLedger:
    def __init__(self, root_dir,repo_id, repo_type, strict=False):
        self.repo_obj = entity.Repo(repo_id=repo_id,repo_type=repo_type)
        self.repo = repo.JSONRepo(root_dir, self.repo_obj, strict)


    def _register(self,filepath: str):
        entity_state = entity.Entity(
            filename=filepath,
            status="pending"
        )
        self.repo.add(entity_state)

    def register_file(self,filepath: str):
        self._register(filepath)

    def register_files(self,filepaths: list[str]):
        for fp in filepaths:
            self._register(fp)
        

    def claim(self):
        # Completed means local output exists but its upload may have been
        # interrupted. The caller resumes that upload before processing again.
        return [
            *self.repo.filter_entity(by="status", value="pending"),
            *self.repo.filter_entity(by="status", value="completed"),
        ]

    def mark_uploaded(self, filepath: str):
        self.repo.update_entity(filepath,{"status":"uploaded"})

    def mark_pending(self, filepath: str):
        self.repo.update_entity(filepath,{"status":"pending", "local_dir": []})

    def mark_completed(self, filepath : str, local_dir_names : list[Path]):
        self.repo.update_entity(filepath,{"status":"completed", "local_dir" : [str(ldn) for ldn in local_dir_names ]})

    def fail(self,filepath):
        self.repo.update_entity(filepath,{"status" : "failed"})

    def get_all_completed(self):
        return self.repo.filter_entity(by="status", value="completed")
    



    def stats(self,):
        #tbd
        pass
