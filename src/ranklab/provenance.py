import hashlib, json, platform
from pathlib import Path
from importlib.metadata import version


def capture(out, source):
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    value = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies={name: version(name) for name in ["numpy", "scipy"]},
        source_sha256=files,
        data=source,
    )
    (Path(out) / "provenance.json").write_text(json.dumps(value, indent=2))
