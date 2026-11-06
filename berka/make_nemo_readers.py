import hashlib
import sys
from pathlib import Path

B = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "berka-saif"
WANT = {"pgu_monitor_stages.py": "4b0b57c668014555b5abd703059113c6a7b640337dab9d16e61e27a52b300959",
        "pgu_procs_reader.py": "17f39c1aadbfc79c005ac12d7ab1ed95df353f9275e17e525ae878129d029588"}
EDITS = {"pgu_procs_reader.py": [('"nemo_seed.sh",', '"nemo_step.sh", "nemo_chain.sh",'),
                                 ('"synthetic-data-bootcamp/.venv/")', '"synthetic-data-bootcamp/.venv/", "/berka-saif/nemo-venv/")')]}
for src, want in WANT.items():
    text = (B / src).read_text()
    got = hashlib.sha256(text.encode()).hexdigest()
    assert got == want, f"{src} is {got}, not {want}"
    text = text.replace("pgu", "nemo")
    for old, new in EDITS.get(src, []):
        assert text.count(old) == 1, f"{old} found {text.count(old)} times in {src}"
        text = text.replace(old, new)
    out = B / src.replace("pgu", "nemo")
    assert not out.exists(), f"{out} exists"
    out.write_text(text)
    print(out.name, hashlib.sha256(text.encode()).hexdigest())
