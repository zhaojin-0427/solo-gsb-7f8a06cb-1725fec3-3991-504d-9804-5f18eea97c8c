"""Build-result provenance attestation verification.

Given the current build graph (same schema as /plans), a *target* node, at
most one signed attestation per node of the target's transitive dependency
closure, and a table of trusted Ed25519 public keys, decide whether every
closure node has an authentic, dependency-consistent attestation.

Attestation digest
------------------
For an attestation ``a`` for node *n*::

    digest(a) = sha256(canonical_json([
        a.node,
        a.fingerprint,
        a.signer,
        [[dep, a.dependency_digests[dep]]
         for dep in sorted(a.dependency_digests)],
    ])).hexdigest()

The canonical bytes are compact JSON (``separators=(",", ":")``) with
non-ASCII emitted literally (``ensure_ascii=False``), encoded as UTF-8 and
with sorted ``[dependency, digest]`` pairs.  The digest is the lowercase
SHA-256 hex value, and the Ed25519 signature covers exactly those bytes.

Verification
------------
Fingerprints over the closure are first recomputed with the /plans rules.
Each closure node must carry exactly one attestation, and for each node the
checks are (first failure wins, in this order):

1. an attestation for the node exists;
2. its claimed fingerprint equals the recomputed fingerprint;
3. for every direct dependency that is named in the attestation: that
   dependency is itself verified successfully, and the claimed digest for
   it equals the recomputed digest of that dependency's attestation;
4. the ``dependency_digests`` keys are exactly the node's direct
   dependencies (an extra key makes this the reported failure only when
   every named dependency already passed step 3);
5. its signer is present in the trusted-key table;
6. its signature verifies under that signer's Ed25519 public key.

Failure of a dependency propagates to every node that (transitively)
depends on it as a ``digest_mismatch``: a claimed proof summary cannot be
trusted when the proof it summarizes is itself untrusted or absent.
Nodes outside the target closure are never scheduled and must not carry
attestations.

Validation
----------
Beyond the usual graph checks (duplicate node, missing dependency, cycle):
unknown target, attestations for nodes outside the closure, attestations
referencing nodes that are not in the graph, more than one attestation for
the same node, and malformed Base64 or wrong-length (not 32 bytes) Ed25519
public keys all produce structured HTTP 400 errors.

Determinism
-----------
All validation and verification goes through name-sorted structures, so
the request payload order never changes the result.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from app.models import (
    Node,
    ProvenanceAttestation,
    ProvenanceReason,
    ProvenanceVerifyResponse,
)
from app.planner import (
    PlanError,
    compute_fingerprints,
    validate_graph,
)

ED25519_KEY_SIZE = 32


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


def _b64decode(value: str) -> bytes | None:
    """Strict Base64 decode; return ``None`` when the encoding is malformed."""
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None


def attestation_bytes(attestation: ProvenanceAttestation) -> bytes:
    """Canonical UTF-8 JSON bytes covered by the attestation signature."""
    pairs = sorted(attestation.dependency_digests.items())
    body = [
        attestation.node,
        attestation.fingerprint,
        attestation.signer,
        [[name, digest] for name, digest in pairs],
    ]
    return json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def attestation_digest(attestation: ProvenanceAttestation) -> str:
    """Lowercase SHA-256 hex of the canonical attestation bytes."""
    return hashlib.sha256(attestation_bytes(attestation)).hexdigest()


def verify_provenance(
    nodes: list[Node],
    target: str,
    attestations: list[ProvenanceAttestation],
    trusted_keys: dict[str, str],
) -> ProvenanceVerifyResponse:
    """Verify the signed attestations covering ``target``'s dependency closure."""
    graph, contents, order = validate_graph(nodes)

    # ---- validation: target resolves -------------------------------------
    if target not in graph:
        raise PlanError(
            "unknown_target",
            f"Target node is not in the graph: {target!r}",
            {"target": target},
        )

    # ---- validation: trusted keys decode to 32-byte Ed25519 keys ----------
    public_keys: dict[str, Ed25519PublicKey] = {}
    malformed: list[str] = []
    wrong_size: list[dict[str, object]] = []
    for signer in sorted(trusted_keys):
        raw = _b64decode(trusted_keys[signer])
        if raw is None:
            malformed.append(signer)
            continue
        if len(raw) != ED25519_KEY_SIZE:
            wrong_size.append({"signer": signer, "length": len(raw)})
            continue
        public_keys[signer] = Ed25519PublicKey.from_public_bytes(raw)
    if malformed:
        raise PlanError(
            "invalid_encoding",
            "Trusted public keys must be valid Base64",
            {"signers": malformed},
        )
    if wrong_size:
        raise PlanError(
            "invalid_public_key",
            f"Ed25519 public keys must decode to exactly {ED25519_KEY_SIZE} bytes",
            {"keys": wrong_size},
        )

    # ---- validation: at most one attestation per node --------------------
    seen_nodes: set[str] = set()
    duplicate_nodes: set[str] = set()
    for attestation in attestations:
        if attestation.node in seen_nodes:
            duplicate_nodes.add(attestation.node)
        seen_nodes.add(attestation.node)
    if duplicate_nodes:
        raise PlanError(
            "duplicate_attestation",
            "At most one attestation may be supplied per node",
            {"nodes": sorted(duplicate_nodes)},
        )

    # ---- validation: attestations reference real nodes -------------------
    unknown_nodes = sorted(
        {a.node for a in attestations if a.node not in graph}
    )
    if unknown_nodes:
        raise PlanError(
            "unknown_attestation_node",
            "Attestations reference nodes that are not in the graph",
            {"names": unknown_nodes},
        )

    closure = _dependency_closure(graph, target)

    # ---- validation: no attestations for nodes outside the closure --------
    outside = sorted(seen_nodes - closure)
    if outside:
        raise PlanError(
            "attestation_outside_closure",
            "Attestations may only cover nodes in the target's dependency closure",
            {"names": outside},
        )

    # ---- validation: every signature is well-formed Base64 ---------------
    bad_signatures = sorted(
        attestation.node
        for attestation in attestations
        if _b64decode(attestation.signature) is None
    )
    if bad_signatures:
        raise PlanError(
            "invalid_encoding",
            "Attestation signatures must be valid Base64",
            {"nodes": bad_signatures},
        )

    # ---- fingerprints (in topological order) and attestation digests ------
    fingerprints = compute_fingerprints(graph, contents, order)
    by_node = {attestation.node: attestation for attestation in attestations}
    digests = {
        name: attestation_digest(attestation)
        for name, attestation in by_node.items()
    }

    # ---- per-node verification in topological order ----------------------
    # A node can only be trusted when every direct dependency is trusted, so
    # processing in topological order lets a failure propagate downstream.
    status: dict[str, str] = {}
    for name in order:
        if name not in closure:
            continue
        attestation = by_node.get(name)
        if attestation is None:
            status[name] = "missing_attestation"
            continue
        if attestation.fingerprint != fingerprints[name]:
            status[name] = "fingerprint_mismatch"
            continue
        deps = set(graph[name])
        claimed = set(attestation.dependency_digests)
        bad_digests = [
            dep
            for dep in sorted(deps & claimed)
            if status.get(dep) is not None
            or attestation.dependency_digests[dep] != digests.get(dep)
        ]
        if bad_digests:
            status[name] = "digest_mismatch"
            continue
        if claimed != deps:
            status[name] = "dependency_mismatch"
            continue
        if attestation.signer not in public_keys:
            status[name] = "unknown_signer"
            continue
        # Base64 validity was checked above; length/content errors surface as
        # an invalid signature rather than a malformed request.
        raw_signature = base64.b64decode(attestation.signature, validate=True)
        try:
            public_keys[attestation.signer].verify(
                raw_signature, attestation_bytes(attestation)
            )
        except InvalidSignature:
            status[name] = "invalid_signature"

    reasons = sorted(
        (
            ProvenanceReason(node=name, code=code)
            for name, code in status.items()
            if code is not None
        ),
        key=lambda reason: reason.node,
    )

    return ProvenanceVerifyResponse(
        trusted=not reasons,
        reasons=reasons,
    )


__all__ = ["attestation_bytes", "attestation_digest", "verify_provenance"]
