#!/usr/bin/env python3
"""Provision an authenticated MedGemma snapshot without exposing credentials.

This script downloads model weights only. It never reads MIMIC-derived ledgers,
notes, tasks, or outputs. A Hugging Face read token is used in memory only.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision a fixed gated MedGemma snapshot locally.")
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--model_id", default="google/medgemma-27b-text-it")
    parser.add_argument("--revision", help="Optional immutable revision. Defaults to the authenticated current commit SHA.")
    parser.add_argument("--access_check_only", action="store_true", help="Verify authenticated gated content access without downloading the model snapshot.")
    parser.add_argument("--prompt_for_token", action="store_true", help="Prompt securely for a token in this provisioning process only.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    # Prefer the shell-scoped token so provisioning never accidentally uses a
    # stale token stored by a prior Hugging Face login.
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token and args.prompt_for_token:
        token = getpass.getpass("Paste Hugging Face read token for this provisioning run: ").strip()
    if not token:
        raise PermissionError("No Hugging Face token is available. Authenticate with a read token after accepting the gated model terms.")
    api = HfApi()
    # This authenticated metadata request confirms that the accepted gated terms
    # and token access are sufficient before any large download starts.
    info = api.model_info(args.model_id, token=token, files_metadata=True)
    revision = args.revision or info.sha
    if not revision:
        raise RuntimeError("Hugging Face did not return an immutable model revision.")
    model_root = Path(args.model_root).resolve()
    # model_info can be visible even when content access has not been granted.
    # Probe a tiny repository file before creating/downloading the full snapshot.
    try:
        hf_hub_download(
            repo_id=args.model_id,
            filename=".gitattributes",
            revision=revision,
            token=token,
            local_dir=str(model_root / ".medgemma_access_probe" / revision),
        )
    except GatedRepoError as exc:
        raise PermissionError(
            "Authenticated content access to the gated MedGemma repository was denied. "
            "Accept the model terms with this same Hugging Face account and use a read token "
            "authorized for google/medgemma-27b-text-it. No model snapshot was downloaded."
        ) from exc
    if args.access_check_only:
        print(json.dumps({"model_id": args.model_id, "revision": revision, "content_access": "confirmed"}, indent=2))
        return
    model_path = model_root / "medgemma-27b-text-it" / revision
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.model_id, revision=revision, local_dir=str(model_path), token=token)

    checksum_path = model_path / "MODEL_SHA256SUMS.txt"
    file_paths = sorted(path for path in model_path.rglob("*") if path.is_file() and path.name != checksum_path.name)
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in file_paths:
            handle.write(f"{sha256(path)}  {path.relative_to(model_path)}\n")
    versions = {"huggingface_hub": huggingface_hub.__version__}
    for module_name in ("transformers", "torch"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not_installed"
    manifest = {
        "model_id": args.model_id,
        "revision": revision,
        "local_path": str(model_path),
        "download_date": datetime.now(timezone.utc).isoformat(),
        "software_versions": versions,
        "platform": platform.platform(),
        "weight_file_count": len([path for path in file_paths if path.suffix in {".safetensors", ".bin"}]),
        "checksum_file": str(checksum_path),
        "security_note": "The Hugging Face read token was used only in memory during provisioning and is not written to this manifest.",
    }
    manifest_path = model_path / "MODEL_PROVISIONING_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("model_id", "revision", "local_path", "checksum_file")}, indent=2))


if __name__ == "__main__":
    main()
