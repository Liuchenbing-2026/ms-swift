"""Relocate immutable checkpoint files to an already mounted storage directory.

Run only after checkpoint writing has finished. Both the host and every reader
container must see the destination at the same relative workspace location.
Copies are SHA256 verified before replacing a source with a relative symlink.
This checks byte preservation, not successful model/optimizer restoration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def persist(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def fingerprint(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def relocate(checkpoint, storage, manifest, filenames, reserve):
    checkpoint = checkpoint.resolve()
    # Preserve the caller's mounted path for relative links inside containers.
    storage = storage.absolute()
    storage.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        raise ValueError('Manifest exists; inspect and resume remaining files explicitly')
    sources = []
    for name in filenames:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Checkpoint-relative file names required')
        source = checkpoint / relative
        if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
            raise ValueError('Expected a regular, singly linked checkpoint file')
        destination = storage / checkpoint.name / relative
        if destination.exists() or destination.with_suffix('.partial').exists():
            raise ValueError('Destination exists; refusing to overwrite')
        sources.append((source, destination))
    required = sum(source.stat().st_size for source, _ in sources)
    if shutil.disk_usage(storage).free < required + reserve:
        raise ValueError('Destination cannot hold the selected files and reserve')
    state = {'status': 'running', 'files': [], 'reserve_bytes': reserve}
    persist(manifest, state)
    try:
        for source, destination in sources:
            before = fingerprint(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.partial')
            state['current'] = str(source)
            state['phase'] = 'copy'
            persist(manifest, state)
            original_hash = hashlib.sha256()
            with source.open('rb') as src, temporary.open('xb') as dst:
                while block := src.read(16 * 1024**2):
                    if shutil.disk_usage(storage).free < reserve + len(block):
                        raise RuntimeError('Destination reserve reached; original retained')
                    dst.write(block)
                    original_hash.update(block)
                dst.flush()
                os.fsync(dst.fileno())
            state['phase'] = 'verify'
            persist(manifest, state)
            copied_hash = hashlib.sha256()
            with temporary.open('rb') as handle:
                while block := handle.read(16 * 1024**2):
                    copied_hash.update(block)
            if copied_hash.digest() != original_hash.digest() or fingerprint(source) != before:
                raise RuntimeError('Hash mismatch or source changed; original retained')
            shutil.copystat(source, temporary)
            temporary.replace(destination)
            fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            record = {'source': str(source), 'destination': str(destination),
                      'bytes': before[2], 'sha256': original_hash.hexdigest(),
                      'copy_verified': True, 'link_installed': False}
            state['files'].append(record)
            persist(manifest, state)
            link = source.with_suffix(source.suffix + '.relocating')
            link.symlink_to(os.path.relpath(destination, source.parent))
            if fingerprint(source) != before:
                raise RuntimeError('Source changed before replacement; original retained')
            os.replace(link, source)
            fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if source.stat().st_size != before[2]:
                raise RuntimeError('Relocated source is not readable')
            record['link_installed'] = True
            state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            persist(manifest, state)
            print(json.dumps(record), flush=True)
        state.update(status='complete', phase='complete', current=None)
        persist(manifest, state)
    except Exception as error:
        state.update(status='failed', error=str(error))
        persist(manifest, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--storage', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--reserve-gib', type=float, default=20)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error('reserve must be nonnegative')
    relocate(args.checkpoint, args.storage, args.manifest, args.files,
             int(args.reserve_gib * 1024**3))
