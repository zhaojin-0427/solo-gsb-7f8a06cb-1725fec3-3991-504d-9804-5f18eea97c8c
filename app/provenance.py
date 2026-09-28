"""Build-result provenance chain verification.

Given the current build graph (same schema as /plans), a release *target*
node, at most one signed attestation per node in the target's dependency
closure and a table of trusted Ed25519 public keys, verify that the whole
build-result provenance chain is authentic.

Attestations and their digests
------------------------------
Each :class:`~app.models.Attestation` names a node and claims:

* that node's recomputed ``fingerprint``;
* the ``signer`` id whose Ed25519 key signed the attestation;
* a mapping from every direct dependency name to that dependency's
  attestation digest.

The attestation's own *digest* is::

    digest = sha256(compact_json([
        node,
        fingerprint,
        signer,
        [[dep_name, dep_digest] for dep_name in sorted(dependencies)],
    ])).hexdigest()

where compact JSON is ``json.dumps(..., separators=(",", ":"),
ensure_ascii=False)`` encoded as UTF-8 (non-ASCII characters are never
escaped).  The Base64-decoded ``signature`` is verified as an Ed25519
signature over exactly those JSON bytes with the signer's trusted public
key.

Verification
------------
The graph fingerprints are recomputed with the /plans rules and the
target's transitive dependency closure is taken.  For every closure node
the service checks, independently, that:

* an attestation exists for the node (``missing_attestation``);
* the signer id is present in the trusted key table (``unknown_signer``);
* the claimed fingerprint equals the recomputed one
  (``fingerprint_mismatch``);
* the dependency mapping names exactly the node's direct dependencies
  (``dependency_mismatch`` for missing/extra names);
* each mapped dependency digest equals the dependency attestation's
  recomputed digest (``dependency_digest_mismatch``);
* the Ed25519 signature verifies (``invalid_signature``).

``trusted`` is true only when none of the above fires.  Failure reasons
are returned per node, sorted by node name; the reason codes inside a
node's list are sorted too, so the request payload order never matters.

Hard errors (HTTP 400)
----------------------
Malformed graphs (duplicate node, missing dependency, cycle), unknown
targets, duplicate attestations for the same closure node, attestations
for nodes outside the closure (including nodes absent from the graph) and
malformed Base64 or wrong key/signature lengths produce structured
``PlanError`` responses instead of a trusted/untrusted verdict.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from app.models import (
    Attestation,
    Node,
    ProvenanceVerifyResponse,
)
from app.planner import (
    PlanError,
    compute_fingerprints,
    validate_graph,
)

PUBLIC_KEY_BYTES = 32
SIGNATURE_BYTES = 64


def _b64decode(value: str) -> bytes:
    """Strictly decode standard Base64; reject URL-safe or trailing junk."""
    return base64.b64decode(value, validate=True)


def _dependency_closure(graph: dict[str, list[str]], target: str) -> set[str]:
    """Target and every node reachable through direct dependencies."""
    closure: set[str] = set()
    stack = [target]
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        closure.add(name)
        stack.extend(graph[name])
    return closure


def attestation_message(attestation: Attestation) -> bytes:
    """Compact, non-ASCII-safe UTF-8 JSON bytes signed by the attestation."""
    pairs = sorted(attestation.dependencies.items())
    body = [
        attestation.node,
        attestation.fingerprint,
        attestation.signer,
        [[name, digest] for name, digest in pairs],
    ]
    return json.dumps(
        body,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def attestation_digest(attestation: Attestation) -> str:
    """Lowercase SHA-256 hex digest of the attestation's canonical message."""
    return hashlib.sha256(attestation_message(attestation)).hexdigest()


def _decode_trusted_keys(trusted_keys: dict[str, str]) -> dict[str, bytes]:
    """Base64-decode every trusted key; malformed encodings raise 400."""
    malformed: list[str] = []
    decoded: dict[str, bytes] = {}
    for signer_id in sorted(trusted_keys):
        try:
            raw_key = _b64decode(trusted_keys[signer_id])
        except (binascii.Error, ValueError):
            malformed.append(signer_id)
            continue
        if len(raw_key) != PUBLIC_KEY_BYTES:
            malformed.append(signer_id)
            continue
        decoded[signer_id] = raw_key
    if malformed:
        raise PlanError(
            "malformed_encoding",
            "Trusted public keys must be standard Base64 of a 32-byte "
            "Ed25519 public key",
            {"trusted_keys": malformed},
        )
    return decoded


def verify_provenance(
    nodes: list[Node],
    target: str,
    attestations: list[Attestation],
    trusted_keys: dict[str, str],
) -> ProvenanceVerifyResponse:
    """Verify the signed provenance chain over the target's dependency closure."""
    graph, contents, order = validate_graph(nodes)

    # ---- validation: target resolves -------------------------------------
    if target not in graph:
        raise PlanError(
            "unknown_target",
            f"Target node is not in the graph: {target!r}",
            {"target": target},
        )

    closure = _dependency_closure(graph, target)

    # ---- validation: attestations only name graph nodes ------------------
    unknown_nodes = sorted({a.node for a in attestations if a.node not in graph})
    if unknown_nodes:
        raise PlanError(
            "unknown_attestation_node",
            "Attestations reference nodes that are not in the graph",
            {"names": unknown_nodes},
        )

    # ---- validation: no attestations outside the target closure ----------
    outside = sorted({a.node for a in attestations if a.node not in closure})
    if outside:
        raise PlanError(
            "attestation_outside_closure",
            "Attestations may only cover nodes in the target's dependency closure",
            {"nodes": outside, "target": target},
        )

    # ---- validation: at most one attestation per closure node ------------
    seen_nodes: set[str] = set()
    duplicate_nodes: set[str] = set()
    for attestation in attestations:
        if attestation.node in seen_nodes:
            duplicate_nodes.add(attestation.node)
        seen_nodes.add(attestation.node)
    if duplicate_nodes:
        raise PlanError(
            "duplicate_attestation",
            "At most one attestation per closure node is allowed",
            {"nodes": sorted(duplicate_nodes)},
        )

    # ---- validation: Base64-encoded keys and signatures ------------------
    key_bytes = _decode_trusted_keys(trusted_keys)
    raw_signatures: dict[str, bytes] = {}
    malformed_signatures: list[str] = []
    for attestation in attestations:
        try:
            raw = _b64decode(attestation.signature)
        except (binascii.Error, ValueError):
            malformed_signatures.append(attestation.node)
            continue
        if len(raw) != SIGNATURE_BYTES:
            malformed_signatures.append(attestation.node)
            continue
        raw_signatures[attestation.node] = raw
    if malformed_signatures:
        raise PlanError(
            "malformed_encoding",
            "Attestation signatures must be standard Base64 of a 64-byte "
            "Ed25519 signature",
            {"signatures": sorted(malformed_signatures)},
        )

    # ---- recompute closure fingerprints ----------------------------------
    fingerprints = compute_fingerprints(graph, contents, order)
    by_node = {attestation.node: attestation for attestation in attestations}
    digests = {
        attestation.node: attestation_digest(attestation)
        for attestation in attestations
    }

    # ---- per-node checks --------------------------------------------------
    reasons: dict[str, list[str]] = {}
    for name in sorted(closure):
        node_reasons: set[str] = set()
        attestation = by_node.get(name)
        if attestation is None:
            reasons[name] = ["missing_attestation"]
            continue

        if attestation.fingerprint != fingerprints[name]:
            node_reasons.add("fingerprint_mismatch")

        claimed_deps = set(attestation.dependencies)
        actual_deps = set(graph[name])
        if claimed_deps != actual_deps:
            node_reasons.add("dependency_mismatch")
        for dep in sorted(claimed_deps & actual_deps):
            if attestation.dependencies[dep] != digests.get(dep):
                node_reasons.add("dependency_digest_mismatch")

        public_bytes = key_bytes.get(attestation.signer)
        if public_bytes is None:
            node_reasons.add("unknown_signer")
        else:
            try:
                Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                    raw_signatures[name],
                    attestation_message(attestation),
                )
            except InvalidSignature:
                node_reasons.add("invalid_signature")

        if node_reasons:
            reasons[name] = sorted(node_reasons)

    return ProvenanceVerifyResponse(trusted=not reasons, reasons=reasons)


__all__ = ["attestation_digest", "attestation_message", "verify_provenance"]
