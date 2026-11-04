import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "nested folder").mkdir()
        (self.source / "nested folder" / "values.txt").write_bytes(b"actual results\n")
        (self.source / "empty.txt").write_bytes(b"")
        self.archive = self.root / "results.tar.gz"

    def write_archive(self, files, manifest=None, extra=None):
        if manifest is None:
            manifest = [{"path": name, "size": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()}
                        for name, data in files]
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, data in files:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            if extra is not None:
                archive.addfile(extra)
            data = json.dumps({"format": 1, "files": manifest}).encode()
            info = tarfile.TarInfo(bundle.MANIFEST)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        self.refresh_sidecar()

    def refresh_sidecar(self):
        bundle.sidecar_path(self.archive).write_text(
            bundle.file_hash(self.archive) + "  " + self.archive.name + "\n",
            encoding="utf-8")

    def test_roundtrip_and_nonoverwrite(self):
        self.assertEqual(bundle.pack(self.source, self.archive), 2)
        self.assertEqual(len(bundle.verify(self.archive)), 2)
        destination = self.root / "downloaded"
        self.assertEqual(bundle.extract(self.archive, destination), 2)
        self.assertEqual((destination / "nested folder" / "values.txt").read_bytes(),
                         b"actual results\n")
        self.assertEqual((destination / "empty.txt").read_bytes(), b"")
        with self.assertRaises(bundle.BundleError):
            bundle.extract(self.archive, destination)
        with self.assertRaises(bundle.BundleError):
            bundle.pack(self.source, self.archive)

    def test_tampered_archive_checksum(self):
        bundle.pack(self.source, self.archive)
        with open(self.archive, "ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(bundle.BundleError, "Archive SHA256"):
            bundle.verify(self.archive)

    def test_wrong_file_hash_with_valid_archive_checksum(self):
        self.write_archive([("result.txt", b"changed")],
                           [{"path": "result.txt", "size": 7, "sha256": "0" * 64}])
        with self.assertRaisesRegex(bundle.BundleError, "File SHA256"):
            bundle.verify(self.archive)

    def test_unsafe_names_rejected_before_extraction(self):
        for name in ("../escape", "/absolute", "C:/drive", "folder/../escape",
                     "folder\\escape", "NUL.txt", "dir/COM1", "COM1 .txt", "trailing.",
                     "folder//file", "data:stream", "folder/space "):
            with self.subTest(name=name):
                self.write_archive([(name, b"x")])
                destination = self.root / "rejected"
                with self.assertRaises(bundle.BundleError):
                    bundle.extract(self.archive, destination)
                self.assertFalse(destination.exists())

    def test_duplicates_case_collisions_and_prefix_collisions(self):
        for files in ([('a', b'1'), ('a', b'2')],
                      [('A.txt', b'1'), ('a.txt', b'2')],
                      [('folder', b'1'), ('folder/file', b'2')]):
            with self.subTest(files=files):
                self.write_archive(files)
                with self.assertRaises(bundle.BundleError):
                    bundle.verify(self.archive)

    def test_unexpected_files_and_links_rejected(self):
        self.write_archive([("extra", b"x")], manifest=[])
        with self.assertRaisesRegex(bundle.BundleError, "manifest"):
            bundle.verify(self.archive)
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                link = tarfile.TarInfo("link")
                link.type = kind
                link.linkname = "../escape"
                self.write_archive([], extra=link)
                with self.assertRaisesRegex(bundle.BundleError, "ordinary files"):
                    bundle.verify(self.archive)

    def test_output_inside_source_is_refused(self):
        with self.assertRaisesRegex(bundle.BundleError, "outside"):
            bundle.pack(self.source, self.source / "output.tar.gz")

    def test_reserved_manifest_name_in_source(self):
        (self.source / bundle.MANIFEST).write_text("collision", encoding="utf-8")
        with self.assertRaisesRegex(bundle.BundleError, "Duplicate"):
            bundle.pack(self.source, self.archive)

    def test_missing_sidecar_is_not_accepted(self):
        self.write_archive([("result.txt", b"value")])
        bundle.sidecar_path(self.archive).unlink()
        with self.assertRaises(OSError):
            bundle.verify(self.archive)

    def test_source_symlink_is_rejected(self):
        outside = self.root / "outside.txt"
        outside.write_text("not in the source", encoding="utf-8")
        try:
            (self.source / "link.txt").symlink_to(outside)
        except OSError as exc:
            self.skipTest("Symlink creation unavailable: " + str(exc))
        with self.assertRaisesRegex(bundle.BundleError, "Links"):
            bundle.pack(self.source, self.archive)


if __name__ == "__main__":
    unittest.main()
