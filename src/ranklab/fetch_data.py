"""Bounded, checksum-verified download; never disables TLS verification."""

import hashlib, json, ssl, urllib.request
from pathlib import Path

ROOT = Path.cwd()


def tls_context():
    context = ssl.create_default_context()
    # Honor the Windows OS trust store as well as Python's bundled roots.
    if hasattr(ssl, "enum_certificates"):
        for certificate, encoding, trust in ssl.enum_certificates("ROOT"):
            if encoding == "x509_asn":
                context.load_verify_locations(
                    cadata=ssl.DER_cert_to_PEM_cert(certificate)
                )
    return context


def fetch():
    source = json.loads(
        Path(__file__).with_name("source.json").read_text(encoding="utf-8")
    )
    path = ROOT / "data/raw" / source["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        path.exists()
        and hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    ):
        print("Verified cached dataset:", path)
        return path
    errors = []
    for url in [source["url"]] + ([source["fallback"]] if "fallback" in source else []):
        try:
            with urllib.request.urlopen(
                url, context=tls_context(), timeout=30
            ) as response:
                data = response.read(40_000_001)
            if len(data) > 40_000_000:
                raise ValueError("Download exceeds 40 MB limit")
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError("Dataset checksum mismatch")
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
            print("Downloaded and verified:", path)
            return path
        except Exception as error:
            errors.append(str(error))
    raise RuntimeError("Dataset download failed: " + "; ".join(errors))


if __name__ == "__main__":
    fetch()
