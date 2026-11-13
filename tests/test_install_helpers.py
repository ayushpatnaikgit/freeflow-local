"""install.sh must replace helper binaries with a new file, never rewrite them in place.

The engine can be running a helper (orukeet-ane) while the installer updates it. Rewriting that file in place makes
macOS kill the next launch of the signed program with SIGKILL (exit 137), which aborted a real update mid-install.
"""

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "install.sh"
SETUP_ANE = ROOT / "setup-ane.sh"


def put_bin_source() -> str:
    match = re.search(r"^put_bin\(\) \{\n.*?^\}\n", INSTALL.read_text(), re.S | re.M)
    if match is None:
        raise AssertionError("install.sh no longer defines put_bin")
    return match.group(0)


def run_put_bin(*args: str) -> None:
    script = put_bin_source() + 'put_bin "$@"\n'
    subprocess.run(["zsh", "-c", script, "put_bin", *args], check=True)


class PutBinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.dest = self.dir / "helper"
        self.dest.write_bytes(b"old build")
        self.dest.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def test_copy_replaces_the_file_instead_of_rewriting_it(self):
        src = self.dir / "src-helper"
        src.write_bytes(b"new build")
        src.chmod(0o755)
        old_inode = self.dest.stat().st_ino
        with open(self.dest, "rb") as running:  # stands in for the engine's running helper
            run_put_bin(str(src), str(self.dest))
            self.assertEqual(running.read(), b"old build", "the running copy must be left untouched")
        self.assertNotEqual(self.dest.stat().st_ino, old_inode, "the helper must be a new file, not rewritten")
        self.assertEqual(self.dest.read_bytes(), b"new build")
        self.assertTrue(os.access(self.dest, os.X_OK))
        self.assertFalse((self.dir / "helper.new").exists(), "no temporary file may be left behind")

    def test_built_variant_renames_the_compiled_file_into_place(self):
        (self.dir / "helper.new").write_bytes(b"compiled build")
        old_inode = self.dest.stat().st_ino
        run_put_bin("--built", str(self.dest))
        self.assertNotEqual(self.dest.stat().st_ino, old_inode)
        self.assertEqual(self.dest.read_bytes(), b"compiled build")
        self.assertFalse((self.dir / "helper.new").exists())


class NoInPlaceHelperWritesTests(unittest.TestCase):
    def test_scripts_never_copy_or_compile_straight_onto_a_helper(self):
        for script in (INSTALL, SETUP_ANE):
            for number, line in enumerate(script.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]
                if re.search(r"\bcp\b.*bin/", code) and "put_bin" not in code:
                    self.fail(f"{script.name}:{number} copies onto a helper in place: {line.strip()}")
                compiled = re.search(r"swiftc\b.*-o\s+\"[^\"]*bin/([\w-]+)(\.new)?\"", code)
                if compiled and not compiled.group(2):
                    self.fail(f"{script.name}:{number} compiles straight onto a helper: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
