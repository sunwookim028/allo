import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).with_name("cleanup_node.py")
    spec = importlib.util.spec_from_file_location("cleanup_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_size_bytes_handles_directory_and_symlink_without_following(tmp_path):
    module = load_module()
    scratch = tmp_path / "scratch"
    nested = scratch / "nested"
    nested.mkdir(parents=True)
    payload = nested / "payload.bin"
    payload.write_bytes(b"payload")
    link = scratch / "payload-link"
    link.symlink_to(payload)

    assert module.size_bytes(scratch) == payload.lstat().st_size + link.lstat().st_size


def test_remove_deletes_directory_tree(tmp_path):
    module = load_module()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "payload.bin").write_bytes(b"payload")

    module.remove(scratch)

    assert not scratch.exists()
