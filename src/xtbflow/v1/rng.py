"""Semantic random keys: batch order and diagnostic calls cannot move streams."""
import hashlib
import json


def addressed_seed(namespace,**keys):
    payload=json.dumps({'namespace':namespace,**keys},sort_keys=True,separators=(',',':'),
                       ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')%(2**63-1)
