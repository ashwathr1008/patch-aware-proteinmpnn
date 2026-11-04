"""Create and verify portable, self-contained PSC result archives (Python 3.9+)."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile


MANIFEST = "__BUNDLE_MANIFEST__.json"
CHUNK = 1024 * 1024
RESERVED = {"CON", "PRN", "AUX", "NUL"}
RESERVED.update(prefix + number for prefix in ("COM", "LPT")
                for number in "123456789\u00b9\u00b2\u00b3")


class BundleError(ValueError):
    pass


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_name(name):
    """Accept the same safe relative file names on both Windows and Linux."""
    if not isinstance(name, str) or not name or name.startswith("/"):
        raise BundleError("Invalid archive path: {!r}".format(name))
    for part in name.split("/"):
        if (part in ("", ".", "..") or part.endswith((".", " "))
                or any(ord(char) < 32 or char in '<>:"\\|?*' for char in part)
                or part.split(".")[0].rstrip(" ").upper() in RESERVED):
            raise BundleError("Unsafe or nonportable path: {!r}".format(name))
    return name


def check_names(names):
    seen = set()
    for name in names:
        key = safe_name(name).casefold()
        if key in seen:
            raise BundleError("Duplicate or case-colliding path: " + name)
        seen.add(key)
    for key in seen:
        parts = key.split("/")
        if any("/".join(parts[:i]) in seen for i in range(1, len(parts))):
            raise BundleError("A file is also used as a directory: " + key)


def regular_stat(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise BundleError("Links and reparse points are not allowed: " + str(path))
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise BundleError("Special files are not allowed: " + str(path))
    return info


def source_files(source):
    files = []
    pending = [source]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir()):
            info = regular_stat(child)
            if stat.S_ISDIR(info.st_mode):
                pending.append(child)
            else:
                files.append(child)
    return sorted(files)


def sidecar_path(archive):
    return archive.with_name(archive.name + ".sha256")


def pack(source, output):
    source = Path(source)
    if not stat.S_ISDIR(regular_stat(source).st_mode):
        raise BundleError("Source must be a directory")
    source = source.resolve()
    output = Path(output).absolute()
    try:
        output.resolve().relative_to(source)
    except ValueError:
        pass
    else:
        raise BundleError("Output must be outside the source directory")
    sidecar = sidecar_path(output)
    if os.path.lexists(output) or os.path.lexists(sidecar):
        raise BundleError("Refusing to overwrite an archive or checksum")
    files = source_files(source)
    names = [path.relative_to(source).as_posix() for path in files]
    check_names(names + [MANIFEST])
    records = []
    with tarfile.open(output, "x:gz", format=tarfile.PAX_FORMAT) as archive:
        for path, name in zip(files, names):
            before = regular_stat(path)
            digest = file_hash(path)
            info = tarfile.TarInfo(name)
            info.size = before.st_size
            info.mode = stat.S_IMODE(before.st_mode) & 0o777
            info.mtime = int(before.st_mtime)
            with open(path, "rb") as handle:
                archive.addfile(info, handle)
            after = regular_stat(path)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise BundleError("Source changed while packing: " + name)
            records.append({"path": name, "size": info.size, "sha256": digest})
        payload = json.dumps({"format": 1, "files": records}, indent=2).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    with open(sidecar, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(file_hash(output) + "  " + output.name + "\n")
    # Detect changes between hashing and archiving before declaring success.
    verify(output)
    return len(records)


def verify(archive_path):
    archive_path = Path(archive_path)
    checksum = sidecar_path(archive_path).read_text(encoding="utf-8").rstrip("\n")
    expected, separator, filename = checksum.partition("  ")
    if (not separator or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or filename != archive_path.name):
        raise BundleError("Invalid checksum sidecar")
    if file_hash(archive_path) != expected:
        raise BundleError("Archive SHA256 mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        check_names([member.name for member in members])
        if any(not member.isfile() for member in members):
            raise BundleError("Only ordinary files are allowed in the archive")
        by_name = {member.name: member for member in members}
        manifest = by_name.pop(MANIFEST, None)
        if manifest is None or manifest.size > 16 * CHUNK:
            raise BundleError("Missing or oversized manifest")
        with archive.extractfile(manifest) as handle:
            metadata = json.load(handle)
        if (not isinstance(metadata, dict) or type(metadata.get("format")) is not int
                or metadata.get("format") != 1
                or not isinstance(metadata.get("files"), list)):
            raise BundleError("Invalid manifest format")
        entries = metadata["files"]
        if any(not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}
               for entry in entries):
            raise BundleError("Invalid file record")
        check_names([entry["path"] for entry in entries] + [MANIFEST])
        if set(by_name) != {entry["path"] for entry in entries}:
            raise BundleError("Archive files do not match the manifest")
        for entry in entries:
            name, size, digest = entry["path"], entry["size"], entry["sha256"]
            if (type(size) is not int or size < 0 or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise BundleError("Invalid size or digest: " + name)
            member = by_name[name]
            if member.size != size:
                raise BundleError("Size mismatch: " + name)
            actual = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for chunk in iter(lambda: handle.read(CHUNK), b""):
                    actual.update(chunk)
            if actual.hexdigest() != digest:
                raise BundleError("File SHA256 mismatch: " + name)
        return entries


def extract(archive_path, destination):
    entries = verify(archive_path)
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise BundleError("Extraction destination must not already exist")
    # Resolve the parent once; never use tarfile.extractall on archive-controlled names.
    destination = destination.parent.resolve(strict=True) / destination.name
    destination.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        for entry in entries:
            target = destination.joinpath(*entry["path"].split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(entry["path"]) as source, open(target, "xb") as output:
                for chunk in iter(lambda: source.read(CHUNK), b""):
                    output.write(chunk)
            if file_hash(target) != entry["sha256"]:
                raise BundleError("Extracted file SHA256 mismatch: " + entry["path"])
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack")
    packing.add_argument("--source", required=True)
    packing.add_argument("--output", required=True)
    for command in ("verify", "extract"):
        checking = commands.add_parser(command)
        checking.add_argument("--archive", required=True)
        if command == "extract":
            checking.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        if args.command == "pack":
            count = pack(args.source, args.output)
        elif args.command == "verify":
            count = len(verify(args.archive))
        else:
            count = extract(args.archive, args.destination)
    except (OSError, ValueError, tarfile.TarError, EOFError) as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
    print("{} successful: {} files".format(args.command, count))


if __name__ == "__main__":
    main()
