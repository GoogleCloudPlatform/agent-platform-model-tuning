# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filesystem abstraction over a published trajectory tree.

Two backends: a local path and a ``gs://`` prefix. Both expose the same
four operations the loader needs, so every layer above is
storage-agnostic and testable against ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Protocol, runtime_checkable

from google.api_core import exceptions as gax

logger = logging.getLogger(__name__)


class TreeReadError(Exception):
    """A read FAILED, as opposed to the object not being there.

    The two are not interchangeable and used to be collapsed into
    ``None``: a transient GCS 503 then looked exactly like a trial with
    no ``trajectory.json``, so a flaky read silently shrank the analyzed
    population and the report said "skipped: missing agent/trajectory.json"
    about a file that exists. Absence stays ``None``; failure raises this,
    and the loader turns it into a skip whose message says so.
    """


@runtime_checkable
class TreeReader(Protocol):
    """Read-only view of a trajectory tree rooted at some prefix."""

    def read_json(self, rel_path: str) -> dict[str, Any] | None: ...
    def read_text(self, rel_path: str) -> str | None: ...
    def list_dir(self, rel_path: str) -> list[str]: ...
    def list_subdirs(self, rel_path: str) -> list[str]: ...
    def exists(self, rel_path: str) -> bool: ...


class LocalReader:
    """TreeReader backed by the local filesystem."""

    def __init__(self, root: str) -> None:
        self._root = pathlib.Path(root)

    def read_text(self, rel_path: str) -> str | None:
        p = self._root / rel_path
        if not p.is_file():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError as exc:
            # Same contract as the GCS side: the file IS there and could
            # not be read, which is not the same as it being absent.
            raise TreeReadError(f"local read failed for {p}: {exc}") from exc

    def read_json(self, rel_path: str) -> dict[str, Any] | None:
        raw = self.read_text(rel_path)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TreeReadError(f"malformed JSON at {rel_path}: {exc}") from exc

    def list_dir(self, rel_path: str) -> list[str]:
        p = self._root / rel_path
        if not p.is_dir():
            return []
        # Sorted to match GcsReader: differing iteration order between
        # backends is a trap for callers.
        return sorted(child.name for child in p.iterdir())

    def list_subdirs(self, rel_path: str) -> list[str]:
        """Child directories only -- the candidate trials, not the files."""
        p = self._root / rel_path
        if not p.is_dir():
            return []
        return sorted(child.name for child in p.iterdir() if child.is_dir())

    def exists(self, rel_path: str) -> bool:
        return (self._root / rel_path).exists()


def parse_gs_url(url: str) -> tuple[str, str]:
    """Split ``gs://bucket/optional/prefix`` into ``(bucket, prefix)``.

    The prefix has no leading or trailing slash. Raises ValueError for a
    non-``gs://`` input.
    """
    if not url.startswith("gs://"):
        raise ValueError(f"Not a gs:// URL: {url!r}")
    rest = url[len("gs://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


class GcsReader:
    """TreeReader backed by a ``gs://`` prefix.

    ``client`` may be injected so the reader is exercisable without a
    network or credentials; when omitted the real storage client is
    constructed lazily.
    """

    def __init__(self, url: str, client: Any | None = None) -> None:
        self._bucket_name, self._prefix = parse_gs_url(url)
        if client is None:
            from google.cloud import storage  # lazy: heavy import

            client = storage.Client()
        self._client = client
        self._bucket = client.bucket(self._bucket_name)

    def _blob_name(self, rel_path: str) -> str:
        # GCS is a flat object store with no path resolution: a literal "."
        # segment or a doubled "/" matches nothing, whereas pathlib drops
        # them, so an unnormalised caller works locally and silently returns
        # NOTHING on GCS -- an empty listing reads as "no trials" rather than
        # as an error. The tree validator in this repo's tooling once gave a
        # run a false clean bill of health exactly this way, so normalise
        # here rather than trusting callers.
        rel = "/".join(seg for seg in rel_path.strip().split("/") if seg and seg != ".")
        parts = [p for p in (self._prefix, rel) if p]
        return "/".join(parts)

    def read_text(self, rel_path: str) -> str | None:
        # One RPC: exists() + download is two round trips and still races.
        name = self._blob_name(rel_path)
        try:
            return self._bucket.blob(name).download_as_text()
        except gax.NotFound:
            return None
        except gax.GoogleAPIError as exc:
            # NOT None: a 503 is not an absent object, and reporting it as
            # one turns a retryable blip into a silently smaller report.
            raise TreeReadError(
                f"GCS read failed for gs://{self._bucket_name}/{name}: {exc}"
            ) from exc

    def read_json(self, rel_path: str) -> dict[str, Any] | None:
        raw = self.read_text(rel_path)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TreeReadError(
                f"malformed JSON at gs://{self._bucket_name}/"
                f"{self._blob_name(rel_path)}: {exc}"
            ) from exc

    def _list_level(self, rel_path: str) -> tuple[set[str], set[str]]:
        """``(files, subdirs)`` directly under ``rel_path``, in one listing.

        ``delimiter="/"`` makes the service do the rollup: it returns the
        objects at this level plus a ``prefixes`` set for the level below,
        instead of every object in the subtree. On a job root with a few
        thousand trials that is the difference between one page and tens
        of thousands of object metadata records.
        """
        # The trailing slash keeps a sibling prefix ("runs/job1-sibling")
        # from being reported as a child of "runs/job1".
        full = self._blob_name(rel_path)
        full = f"{full}/" if full else ""
        iterator = self._client.list_blobs(
            self._bucket_name, prefix=full, delimiter="/"
        )
        files: set[str] = set()
        # The iterator must be drained BEFORE ``prefixes`` is read: the
        # client populates it page by page as the listing is consumed.
        for blob in iterator:
            tail = blob.name[len(full) :]
            if tail and "/" not in tail:
                files.add(tail)
        subdirs = {
            p[len(full) :].strip("/")
            for p in (getattr(iterator, "prefixes", None) or ())
        }
        subdirs.discard("")
        return files, subdirs

    def list_dir(self, rel_path: str) -> list[str]:
        files, subdirs = self._list_level(rel_path)
        return sorted(files | subdirs)

    def list_subdirs(self, rel_path: str) -> list[str]:
        """Child directories only -- the candidate trials, not the files."""
        return sorted(self._list_level(rel_path)[1])

    def exists(self, rel_path: str) -> bool:
        name = self._blob_name(rel_path)
        if self._bucket.blob(name).exists():
            return True
        # Directories are not objects on GCS, but LocalReader reports True
        # for one, so fall back to a cheap prefix probe to keep the two
        # backends in agreement.
        prefix = f"{name}/" if name else ""
        listing = self._client.list_blobs(
            self._bucket_name, prefix=prefix, max_results=1
        )
        return any(True for _ in listing)


def make_reader(root: str) -> TreeReader:
    """Return the right reader for a local path or a ``gs://`` URL."""
    return GcsReader(root) if root.startswith("gs://") else LocalReader(root)
