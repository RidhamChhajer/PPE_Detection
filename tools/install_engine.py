"""Publish a verified engine/metadata pair with backup and fail-closed readers."""
from pathlib import Path
import os
import shutil
import uuid

from app.detector import model_specs
from tools.model_run import sha256


def install_engine(candidate, metadata, models, backup_root):
    candidate, metadata, models = Path(candidate), Path(metadata), Path(models)
    spec = next(s for s in model_specs(candidate.parent) if s.path == candidate.resolve())
    if metadata != candidate.with_suffix('.toml'):
        raise ValueError('Candidate engine and metadata must be paired')
    backup = Path(backup_root)/uuid.uuid4().hex
    backup.mkdir(parents=True)
    models.mkdir(parents=True, exist_ok=True)
    lock = models/'.installing'
    # Exclusive marker prevents competing installers; runtime refuses discovery
    # during the two file replacements (OSes cannot atomically replace a pair).
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    targets = [models/candidate.name, models/metadata.name]
    staged = []
    replaced = []
    release_lock = True
    try:
        for source, target in zip((candidate, metadata), targets):
            if target.exists():
                shutil.copy2(target, backup/target.name)
            temporary = target.with_name(target.name + '.pending')
            shutil.copyfile(source, temporary)
            staged.append(temporary)
        if sha256(staged[0]) != spec.engine_sha256:
            raise RuntimeError('Staged engine checksum mismatch')
        for temporary, target in zip(staged, targets):
            os.replace(temporary, target)
            replaced.append(target)
    except BaseException:
        try:
            for target in reversed(replaced):
                prior = backup/target.name
                if prior.exists():
                    shutil.copy2(prior, target)
                else:
                    target.unlink(missing_ok=True)
        except BaseException:
            # Never expose a potentially mixed pair after a failed rollback.
            # Preserve the backup and marker for explicit recovery.
            release_lock = False
            raise
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
        if release_lock:
            lock.unlink(missing_ok=True)
    return backup
