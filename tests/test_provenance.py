"""Tests for the /provenance/verify endpoint and its verification core."""
from __future__ import annotations

import base64
import hashlib
import json
import random

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from app.main import app
from app.models import Node, ProvenanceAttestation
from app.planner import PlanError, compute_fingerprints, validate_graph
from app.provenance import attestation_bytes, attestation_digest, verify_provenance

client = TestClient(app)


# ---------------------------------------------------------------- helpers


def node(name: str, content: str, deps: list[str] | None = None) -> Node:
    return Node(name=name, content_digest=content, dependency_names=deps or [])


def readme_nodes() -> list[Node]:
    return [
        node("core", "core-src-v1"),
        node("util", "util-src-v1", ["core"]),
        node("api", "api-src-v1", ["util", "core"]),
        node("web", "web-src-v1", ["api"]),
        node("docs", "docs-src-v1"),
    ]


def fingerprints_of(nodes: list[Node]) -> dict[str, str]:
    graph, contents, order = validate_graph(nodes)
    return compute_fingerprints(graph, contents, order)


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def raw_public(private: Ed25519PrivateKey) -> str:
    return b64(
        private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    )


class Signer:
    """Test-side signer that produces canonical, valid attestations."""

    def __init__(self, signer_id: str = "builder-1") -> None:
        self.id = signer_id
        self.private = Ed25519PrivateKey.generate()
        self.public = raw_public(self.private)

    def attest(
        self,
        name: str,
        fingerprint: str,
        dependency_digests: dict[str, str],
        signer_id: str | None = None,
        key: Ed25519PrivateKey | None = None,
    ) -> ProvenanceAttestation:
        signer_id = signer_id or self.id
        unsigned = ProvenanceAttestation(
            node=name,
            fingerprint=fingerprint,
            signer=signer_id,
            dependency_digests=dependency_digests,
            signature="placeholder",
        )
        private = key or self.private
        signature = b64(private.sign(attestation_bytes(unsigned)))
        return ProvenanceAttestation(
            node=name,
            fingerprint=fingerprint,
            signer=signer_id,
            dependency_digests=dependency_digests,
            signature=signature,
        )


def sign_chain(
    nodes: list[Node],
    closure_names: list[str],
    signer: Signer,
    fps: dict[str, str] | None = None,
) -> tuple[list[ProvenanceAttestation], dict[str, str]]:
    """Sign closure nodes in topological order; return (attestations, digests)."""
    graph, _, order = validate_graph(nodes)
    fps = fps or fingerprints_of(nodes)
    digests: dict[str, str] = {}
    by_name: dict[str, ProvenanceAttestation] = {}
    wanted = set(closure_names)
    for name in order:
        if name not in wanted:
            continue
        dep_digests = {dep: digests[dep] for dep in sorted(graph[name])}
        attestation = signer.attest(name, fps[name], dep_digests)
        by_name[name] = attestation
        digests[name] = attestation_digest(attestation)
    return [by_name[name] for name in sorted(by_name)], digests


def as_payload(attestations: list[ProvenanceAttestation]) -> list[dict]:
    return [a.model_dump() for a in attestations]


# --------------------------------------------------- canonical attestation


def test_attestation_bytes_compact_utf8_sorted_pairs():
    attestation = ProvenanceAttestation(
        node="nöd-中",
        fingerprint="fp",
        signer="s",
        dependency_digests={"z": "d2", "a": "d1"},
        signature="ignored",
    )
    assert attestation_bytes(attestation) == (
        '["nöd-中","fp","s",[["a","d1"],["z","d2"]]]'
    ).encode("utf-8")


def test_attestation_digest_is_lowercase_sha256_hex():
    attestation = ProvenanceAttestation(
        node="n",
        fingerprint="fp",
        signer="s",
        dependency_digests={"a": "d1"},
        signature="ignored",
    )
    expected = hashlib.sha256(attestation_bytes(attestation)).hexdigest()
    assert attestation_digest(attestation) == expected
    assert attestation_digest(attestation) == attestation_digest(attestation).lower()
    assert len(attestation_digest(attestation)) == 64


def test_digest_independent_of_dependency_map_order():
    a1 = ProvenanceAttestation(
        node="n", fingerprint="fp", signer="s",
        dependency_digests={"a": "d1", "b": "d2"}, signature="x",
    )
    a2 = ProvenanceAttestation(
        node="n", fingerprint="fp", signer="s",
        dependency_digests={"b": "d2", "a": "d1"}, signature="y",
    )
    assert attestation_digest(a1) == attestation_digest(a2)


def test_digest_covers_every_claim_field():
    base = dict(
        node="n", fingerprint="fp", signer="s",
        dependency_digests={"a": "d1"}, signature="x",
    )
    digest = attestation_digest(ProvenanceAttestation(**base))
    for change in (
        dict(node="other"),
        dict(fingerprint="other"),
        dict(signer="other"),
        dict(dependency_digests={"a": "other"}),
        dict(dependency_digests={"b": "d1"}),
        dict(dependency_digests={}),
    ):
        mutated = {**base, **change}
        assert attestation_digest(ProvenanceAttestation(**mutated)) != digest


def test_signature_field_not_part_of_digest():
    a1 = ProvenanceAttestation(
        node="n", fingerprint="fp", signer="s",
        dependency_digests={"a": "d1"}, signature="sig-1",
    )
    a2 = a1.model_copy(update={"signature": "sig-2"})
    assert attestation_digest(a1) == attestation_digest(a2)


# -------------------------------------------------------------- happy path


def test_verifies_full_closure_signed_by_trusted_key():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    assert result.trusted is True
    assert result.reasons == []


def test_target_alone_with_no_dependencies():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestation = signer.attest("docs", fps["docs"], {})
    result = verify_provenance(
        nodes, "docs", [attestation], {signer.id: signer.public}
    )
    assert result.trusted is True
    assert result.reasons == []


def test_different_signers_per_node_all_trusted():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    graph, _, order = validate_graph(nodes)
    signers = {name: Signer(f"builder-{name}") for name in ("core", "util", "api", "web")}
    keys = {s.id: s.public for s in signers.values()}
    digests: dict[str, str] = {}
    attestations: list[ProvenanceAttestation] = []
    for name in order:
        if name not in signers:
            continue
        dep_digests = {dep: digests[dep] for dep in sorted(graph[name])}
        a = signers[name].attest(name, fps[name], dep_digests)
        digests[name] = attestation_digest(a)
        attestations.append(a)
    result = verify_provenance(nodes, "web", attestations, keys)
    assert result.trusted is True


def test_node_outside_closure_needs_no_attestation():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # docs is not in web's closure; supplying nothing is fine.
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    assert result.trusted is True


# -------------------------------------------------------- failure reasons


def test_missing_attestation_for_leaf_reports_node_and_propagates():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    present = [a for a in attestations if a.node != "core"]
    result = verify_provenance(
        nodes, "web", present, {signer.id: signer.public}
    )
    assert result.trusted is False
    codes = {r.node: r.code for r in result.reasons}
    assert codes["core"] == "missing_attestation"
    # Every downstream node relies on core's (absent) proof summary.
    assert codes["util"] == "digest_mismatch"
    assert codes["api"] == "digest_mismatch"
    assert codes["web"] == "digest_mismatch"
    assert [r.node for r in result.reasons] == sorted(codes)


def test_missing_attestation_for_target_only():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api"], signer, fps
    )
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes == {"web": "missing_attestation"}


def test_fingerprint_mismatch_reports_node_and_propagates():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, digests = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # Re-sign util with a wrong claimed fingerprint (a valid signature over
    # the wrong content).
    replaced = signer.attest(
        "util", "0" * 64, {"core": digests["core"]}
    )
    attestations = [a if a.node != "util" else replaced for a in attestations]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["util"] == "fingerprint_mismatch"
    assert codes["api"] == "digest_mismatch"
    assert codes["web"] == "digest_mismatch"
    assert "core" not in codes


def test_dependency_digest_mismatch_reports_node_and_propagates():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, digests = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    replaced = signer.attest(
        "util", fps["util"], {"core": "f" * 64}
    )
    attestations = [a if a.node != "util" else replaced for a in attestations]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["util"] == "digest_mismatch"
    assert codes["api"] == "digest_mismatch"
    assert codes["web"] == "digest_mismatch"
    assert "core" not in codes


def test_dependency_name_set_mismatch_missing_key():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, digests = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # util lists no dependency digests at all, so its (missing) core digest
    # cannot be checked: key-set mismatch surfaces first.
    replaced = signer.attest("util", fps["util"], {})
    attestations = [a if a.node != "util" else replaced for a in attestations]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["util"] == "dependency_mismatch"


def test_dependency_name_set_mismatch_extra_key():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, digests = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # util lists a digest for docs, which is not its direct dependency.
    replaced = signer.attest(
        "util", fps["util"],
        {"core": digests["core"], "docs": "d" * 64},
    )
    attestations = [a if a.node != "util" else replaced for a in attestations]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["util"] == "dependency_mismatch"
    assert codes["api"] == "digest_mismatch"
    assert codes["web"] == "digest_mismatch"


def test_unknown_signer_reports_node_and_propagates():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, digests = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    replaced = signer.attest(
        "util", fps["util"], {"core": digests["core"]}, signer_id="stranger"
    )
    attestations = [a if a.node != "util" else replaced for a in attestations]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["util"] == "unknown_signer"
    assert codes["api"] == "digest_mismatch"
    assert codes["web"] == "digest_mismatch"


def test_signature_by_other_private_key_is_invalid():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    other = Signer("other")
    attestations, digests = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # Claim signer.id but sign with a different private key.
    replaced = signer.attest(
        "util", fps["util"], {"core": digests["core"]}, key=other.private
    )
    attestations = [a if a.node != "util" else replaced for a in attestations]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["util"] == "invalid_signature"
    assert codes["api"] == "digest_mismatch"
    assert codes["web"] == "digest_mismatch"


def test_tampered_signature_bytes_is_invalid():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    by_name = {a.node: a for a in attestations}
    tampered = bytearray(base64.b64decode(by_name["core"].signature))
    tampered[0] ^= 0x01
    by_name["core"] = by_name["core"].model_copy(
        update={"signature": b64(bytes(tampered))}
    )
    attestations = [by_name[name] for name in sorted(by_name)]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["core"] == "invalid_signature"
    assert codes["util"] == "digest_mismatch"
    assert codes["api"] == "digest_mismatch"
    assert codes["web"] == "digest_mismatch"


def test_signature_wrong_length_is_invalid_signature_not_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    by_name = {a.node: a for a in attestations}
    by_name["core"] = by_name["core"].model_copy(
        update={"signature": b64(b"x" * 63)}
    )
    attestations = [by_name[name] for name in sorted(by_name)]
    result = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    assert codes["core"] == "invalid_signature"
    assert codes["util"] == "digest_mismatch"


def test_reasons_sorted_by_node_name():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # Drop everything except web: all nodes are untrusted.
    result = verify_provenance(
        nodes, "web",
        [a for a in attestations if a.node == "web"],
        {signer.id: signer.public},
    )
    assert [r.node for r in result.reasons] == ["api", "core", "util", "web"]


def test_first_failure_per_node_takes_precedence():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    # A fully broken attestation for the target: wrong fingerprint AND no
    # dependency digests AND unknown signer AND garbage signature.
    broken = ProvenanceAttestation(
        node="web",
        fingerprint="deadbeef",
        signer="nobody",
        dependency_digests={},
        signature=b64(b"0" * 64),
    )
    result = verify_provenance(
        nodes, "web", [broken], {signer.id: signer.public}
    )
    codes = {r.node: r.code for r in result.reasons}
    # Fingerprint mismatch precedes every other check.
    assert codes["web"] == "fingerprint_mismatch"


def test_untrusted_key_in_table_cannot_verify():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer("real")
    impersonator = Signer("real")  # different key under the same id
    attestations, digests = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    result = verify_provenance(
        nodes, "web", attestations, {"real": impersonator.public}
    )
    assert result.trusted is False
    assert all(r.code == "invalid_signature" or r.code == "digest_mismatch"
               for r in result.reasons)
    assert {r.node for r in result.reasons} == {"core", "util", "api", "web"}


# ----------------------------------------------------------- 400 validation


@pytest.mark.parametrize(
    "mutated_nodes,code",
    [
        # duplicate node name
        (readme_nodes() + [node("web", "other")], "duplicate_node"),
        # missing dependency
        ([node("a", "a", ["ghost"])], "missing_dependency"),
        # cycle
        (
            [node("a", "a", ["b"]), node("b", "b", ["a"])],
            "cycle_detected",
        ),
    ],
)
def test_invalid_graph_propagates_plan_error(mutated_nodes, code):
    signer = Signer()
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(mutated_nodes, "web", [], {signer.id: signer.public})
    assert excinfo.value.code == code


def test_unknown_target_is_400():
    nodes = readme_nodes()
    signer = Signer()
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(nodes, "ghost", [], {signer.id: signer.public})
    assert excinfo.value.code == "unknown_target"
    assert excinfo.value.details == {"target": "ghost"}


def test_duplicate_attestation_for_same_node_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestation = signer.attest("docs", fps["docs"], {})
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(
            nodes, "docs", [attestation, attestation],
            {signer.id: signer.public},
        )
    assert excinfo.value.code == "duplicate_attestation"
    assert excinfo.value.details == {"nodes": ["docs"]}


def test_attestation_for_unknown_node_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    ghost = signer.attest("ghost", "x", {})
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(
            nodes, "docs", [ghost], {signer.id: signer.public}
        )
    assert excinfo.value.code == "unknown_attestation_node"
    assert excinfo.value.details == {"names": ["ghost"]}


def test_attestation_outside_closure_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    extra = signer.attest("docs", fps["docs"], {})
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(
            nodes, "web", attestations + [extra],
            {signer.id: signer.public},
        )
    assert excinfo.value.code == "attestation_outside_closure"
    assert excinfo.value.details == {"names": ["docs"]}


def test_malformed_base64_public_key_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestation = signer.attest("docs", fps["docs"], {})
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(nodes, "docs", [attestation], {"builder-1": "@@@"})
    assert excinfo.value.code == "invalid_encoding"
    assert excinfo.value.details == {"signers": ["builder-1"]}


def test_wrong_length_public_key_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestation = signer.attest("docs", fps["docs"], {})
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(
            nodes, "docs", [attestation],
            {"builder-1": b64(b"not-32-bytes")},
        )
    assert excinfo.value.code == "invalid_public_key"
    assert excinfo.value.details == {
        "keys": [{"signer": "builder-1", "length": 12}]
    }


def test_malformed_base64_signature_is_400():
    nodes = readme_nodes()
    signer = Signer()
    broken = ProvenanceAttestation(
        node="docs", fingerprint="x", signer=signer.id,
        dependency_digests={}, signature="not base64!!!",
    )
    with pytest.raises(PlanError) as excinfo:
        verify_provenance(
            nodes, "docs", [broken], {signer.id: signer.public}
        )
    assert excinfo.value.code == "invalid_encoding"
    assert excinfo.value.details == {"nodes": ["docs"]}


# -------------------------------------------------------- determinism


def test_input_order_does_not_change_result():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    baseline = verify_provenance(
        nodes, "web", attestations, {signer.id: signer.public}
    ).model_dump()

    rng = random.Random(1725)
    for _ in range(10):
        shuffled_nodes = nodes[:]
        rng.shuffle(shuffled_nodes)
        shuffled_attestations = attestations[:]
        rng.shuffle(shuffled_attestations)
        keys = dict(list({signer.id: signer.public}.items()))
        result = verify_provenance(
            shuffled_nodes, "web", shuffled_attestations, keys
        ).model_dump()
        assert result == baseline


def test_failure_result_order_independent():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # Drop core and web -> every remaining node is untrusted.
    sparse = [a for a in attestations if a.node in {"util", "api"}]
    first = verify_provenance(
        nodes, "web", sparse, {signer.id: signer.public}
    ).model_dump()
    rng = random.Random(42)
    for _ in range(10):
        shuffled = sparse[:]
        rng.shuffle(shuffled)
        again = verify_provenance(
            nodes, "web", shuffled, {signer.id: signer.public}
        ).model_dump()
        assert again == first
    assert [r["node"] for r in first["reasons"]] == sorted(
        r["node"] for r in first["reasons"]
    )


# ------------------------------------------------------------------ HTTP


def _http_body(
    nodes: list[Node],
    target: str,
    attestations: list[ProvenanceAttestation],
    keys: dict[str, str],
) -> dict:
    return {
        "nodes": [n.model_dump() for n in nodes],
        "target": target,
        "attestations": as_payload(attestations),
        "trusted_keys": keys,
    }


def test_http_success():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    response = client.post(
        "/provenance/verify",
        json=_http_body(
            nodes, "web", attestations, {signer.id: signer.public}
        ),
    )
    assert response.status_code == 200
    assert response.json() == {"trusted": True, "reasons": []}


def test_http_untrusted_returns_200_with_reasons():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    # Tamper core's signature; every downstream proof chain breaks.
    by_name = {a.node: a for a in attestations}
    tampered = bytearray(base64.b64decode(by_name["core"].signature))
    tampered[0] ^= 0xFF
    by_name["core"] = by_name["core"].model_copy(
        update={"signature": b64(bytes(tampered))}
    )
    attestations = [by_name[name] for name in sorted(by_name)]
    response = client.post(
        "/provenance/verify",
        json=_http_body(
            nodes, "web", attestations, {signer.id: signer.public}
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["trusted"] is False
    codes = {r["node"]: r["code"] for r in body["reasons"]}
    assert codes == {
        "core": "invalid_signature",
        "util": "digest_mismatch",
        "api": "digest_mismatch",
        "web": "digest_mismatch",
    }


def test_http_unknown_target_is_400():
    nodes = readme_nodes()
    signer = Signer()
    response = client.post(
        "/provenance/verify",
        json=_http_body(nodes, "ghost", [], {signer.id: signer.public}),
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "unknown_target"


def test_http_outside_closure_attestation_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestations, _ = sign_chain(
        nodes, ["core", "util", "api", "web"], signer, fps
    )
    extra = signer.attest("docs", fps["docs"], {})
    response = client.post(
        "/provenance/verify",
        json=_http_body(
            nodes, "web", attestations + [extra],
            {signer.id: signer.public},
        ),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "attestation_outside_closure"


def test_http_duplicate_attestation_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestation = signer.attest("docs", fps["docs"], {})
    response = client.post(
        "/provenance/verify",
        json=_http_body(
            nodes, "docs", [attestation, attestation],
            {signer.id: signer.public},
        ),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "duplicate_attestation"


def test_http_malformed_public_key_encoding_is_400():
    nodes = readme_nodes()
    fps = fingerprints_of(nodes)
    signer = Signer()
    attestation = signer.attest("docs", fps["docs"], {})
    response = client.post(
        "/provenance/verify",
        json=_http_body(nodes, "docs", [attestation], {signer.id: "@@@"}),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_encoding"


def test_http_graph_cycle_is_400():
    nodes = [node("a", "a", ["b"]), node("b", "b", ["a"])]
    signer = Signer()
    response = client.post(
        "/provenance/verify",
        json=_http_body(nodes, "a", [], {signer.id: signer.public}),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "cycle_detected"


def test_http_unknown_field_is_422():
    nodes = readme_nodes()
    signer = Signer()
    body = _http_body(nodes, "docs", [], {signer.id: signer.public})
    body["bogus"] = 1
    response = client.post("/provenance/verify", json=body)
    assert response.status_code == 422


def test_http_missing_required_field_is_422():
    response = client.post("/provenance/verify", json={"nodes": []})
    assert response.status_code == 422
