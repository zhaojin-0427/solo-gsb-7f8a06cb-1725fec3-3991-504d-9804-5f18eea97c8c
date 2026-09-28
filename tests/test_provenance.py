"""Tests for the /provenance/verify endpoint and its verification core."""
from __future__ import annotations

import base64
import hashlib
import json
import random

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from fastapi.testclient import TestClient

from app.main import app
from app.models import Attestation, Node
from app.planner import PlanError
from app.provenance import (
    attestation_digest,
    attestation_message,
    verify_provenance,
)

client = TestClient(app)


# ---------------------------------------------------------------- helpers

SIGNER_1 = "builder-1"
SIGNER_2 = "builder-2"
UNTRUSTED = "outsider"

# Fixed 32-byte seeds so the tests (and README examples) are reproducible.
SEED_1 = base64.b64decode("mWF1S/Qsd1Z1A2RE4tQFeDGofUyAaVdoAPQB6UBUnUA=")
SEED_2 = b"provenance-signer-two-seed-12345"[:32]
SEED_3 = b"untrusted-signer-seed-1234567890"[:32]
assert len(SEED_2) == len(SEED_3) == 32


def _priv(seed: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(seed)


KEY_1 = _priv(SEED_1)
KEY_2 = _priv(SEED_2)
KEY_3 = _priv(SEED_3)


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def pub_b64(key: Ed25519PrivateKey) -> str:
    return b64(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


TRUSTED_KEYS = {
    SIGNER_1: pub_b64(KEY_1),
    SIGNER_2: pub_b64(KEY_2),
}


def node(name: str, content: str, deps: list[str] | None = None) -> Node:
    return Node(name=name, content_digest=content, dependency_names=deps or [])


def chain_nodes() -> list[Node]:
    # docs is deliberately outside util's closure.
    return [
        node("core", "core-src-v1"),
        node("util", "util-src-v1", ["core"]),
        node("docs", "docs-src-v1"),
    ]


def graph_fingerprint(content: str, deps: list[tuple[str, str]]) -> str:
    """Independent copy of the /plans fingerprint rule for test expectations."""
    body = [content, [[name, fp] for name, fp in sorted(deps)]]
    raw = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


F_CORE = graph_fingerprint("core-src-v1", [])
F_UTIL = graph_fingerprint("util-src-v1", [("core", F_CORE)])


def make_attestation(
    att_node: str,
    fingerprint: str,
    dep_digests: dict[str, str],
    signer: str = SIGNER_1,
    signing_key: Ed25519PrivateKey | None = None,
) -> Attestation:
    """Build and sign an attestation over the canonical message bytes."""
    key = signing_key if signing_key is not None else KEY_1
    model = Attestation.model_construct(
        node=att_node,
        fingerprint=fingerprint,
        signer=signer,
        dependencies=dict(dep_digests),
        signature="",
    )
    signature = b64(key.sign(attestation_message(model)))
    return Attestation(
        node=att_node,
        fingerprint=fingerprint,
        signer=signer,
        dependencies=dict(dep_digests),
        signature=signature,
    )


def signed_chain(
    core_key: Ed25519PrivateKey = KEY_1,
    util_key: Ed25519PrivateKey = KEY_1,
    core_signer: str = SIGNER_1,
    util_signer: str = SIGNER_1,
    core_fp: str = F_CORE,
    util_fp: str = F_UTIL,
) -> list[Attestation]:
    core_att = make_attestation("core", core_fp, {}, core_signer, core_key)
    util_att = make_attestation(
        "util",
        util_fp,
        {"core": attestation_digest(core_att)},
        util_signer,
        util_key,
    )
    return [core_att, util_att]


def verify(
    attestations: list[Attestation],
    *,
    target: str = "util",
    nodes: list[Node] | None = None,
    trusted: dict[str, str] | None = None,
):
    return verify_provenance(
        nodes if nodes is not None else chain_nodes(),
        target,
        attestations,
        trusted if trusted is not None else dict(TRUSTED_KEYS),
    )


# ---------------------------------------------------------------- golden

# The golden vector was generated offline; pinning it locks the canonical
# serialization, digest and Ed25519 signature semantics.
GOLDEN_PUB = "kegzJbcb8pwgmbAFCH74+A1gvtw+MJImy+FxRK/RxAw="
GOLDEN_CORE_DIGEST = "35fda4461d08d70d865b13d5c63d4a1cba1bf5a00dabf651e42a89464e3ec4e6"
GOLDEN_CORE_SIG = (
    "FIuHzVQUY5GursLAKIinbwak0sDffPFx4D4GznlZx6BBMtPZkVYKquvEopQWSu39"
    "y6BPHtzBKSYE1nOATAWqBA=="
)
GOLDEN_UTIL_DIGEST = "ea88a79a0e9f526bfde053fb486642735c0b35e5bf30960ba91aed82892657a2"
GOLDEN_UTIL_SIG = (
    "RLMEy8ak2JPrxdUh7x0BXVU1mkaQiLpXJTeX+sWewMj60FQxwjFgU28AjWBPu8f4"
    "n7qr960/Wv8AgjRVOSxPDg=="
)


def test_golden_key_matches_seed():
    assert pub_b64(KEY_1) == GOLDEN_PUB


def test_golden_core_vector():
    att = Attestation(
        node="core",
        fingerprint=F_CORE,
        signer=SIGNER_1,
        dependencies={},
        signature=GOLDEN_CORE_SIG,
    )
    assert attestation_digest(att) == GOLDEN_CORE_DIGEST
    Ed25519PublicKey.from_public_bytes(base64.b64decode(GOLDEN_PUB)).verify(
        base64.b64decode(GOLDEN_CORE_SIG),
        attestation_message(att),
    )


def test_golden_util_vector():
    att = Attestation(
        node="util",
        fingerprint=F_UTIL,
        signer=SIGNER_1,
        dependencies={"core": GOLDEN_CORE_DIGEST},
        signature=GOLDEN_UTIL_SIG,
    )
    assert attestation_digest(att) == GOLDEN_UTIL_DIGEST


# ------------------------------------------------------------- trusted


def test_valid_chain_is_trusted():
    result = verify(signed_chain())
    assert result.trusted is True
    assert result.reasons == {}


def test_two_trusted_signers_chain_verifies():
    result = verify(signed_chain(util_key=KEY_2, util_signer=SIGNER_2))
    assert result.trusted is True


def test_target_itself_is_in_closure():
    result = verify([make_attestation("core", F_CORE, {})], target="core")
    assert result.trusted is True


def test_recomputed_fingerprints_match_plans_rule():
    assert F_CORE == "b53b764e740673cd4449f872a6cdea0998cbd5d92d2d706820ec010a56e0a714"
    assert F_UTIL == "30c69063e340908c43750e92d696cc9410c09814fef4510c05829ceadc7ca3b8"


def test_input_order_does_not_matter():
    attestations = signed_chain()
    baseline = verify(attestations).model_dump()
    rng = random.Random(20260928)
    for _ in range(10):
        shuffled_nodes = chain_nodes()
        rng.shuffle(shuffled_nodes)
        shuffled = list(attestations)
        rng.shuffle(shuffled)
        result = verify(shuffled, nodes=shuffled_nodes).model_dump()
        assert result == baseline


def test_dependency_map_input_order_does_not_matter():
    # Three deps; reordering the JSON mapping and re-signing must still verify,
    # and the canonical message always lists pairs sorted by dependency name.
    nodes = [
        node("a", "a"),
        node("b", "b"),
        node("c", "c"),
        node("t", "t", ["a", "b", "c"]),
    ]
    f_a = graph_fingerprint("a", [])
    f_b = graph_fingerprint("b", [])
    f_c = graph_fingerprint("c", [])
    f_t = graph_fingerprint("t", [("a", f_a), ("b", f_b), ("c", f_c)])
    att_a = make_attestation("a", f_a, {})
    att_b = make_attestation("b", f_b, {})
    att_c = make_attestation("c", f_c, {})
    mapping = {
        "c": attestation_digest(att_c),
        "a": attestation_digest(att_a),
        "b": attestation_digest(att_b),
    }
    att_t = make_attestation("t", f_t, mapping)
    result = verify([att_t, att_b, att_c, att_a], target="t", nodes=nodes)
    assert result.trusted is True
    message = attestation_message(att_t).decode()
    assert message.index('"a"') < message.index('"b"') < message.index('"c"')


def test_non_ascii_is_not_escaped():
    # Node name/content/signer with non-ASCII characters: ensure_ascii=False
    # means the signed bytes carry literal UTF-8.
    nodes = [node("café", "é-content-ζ")]
    fingerprint = graph_fingerprint("é-content-ζ", [])
    att = make_attestation("café", fingerprint, {}, "签署者α")
    message = attestation_message(att)
    assert b"caf\xc3\xa9" in message
    assert b"\xe7\xad\xbe\xe7\xbd\xb2\xe8\x80\x85\xce\xb1" in message
    assert b"\\u" not in message
    # Independently recompute the digest from the raw UTF-8 document.
    expected = hashlib.sha256(
        json.dumps(
            ["café", fingerprint, "签署者α", []],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert attestation_digest(att) == expected
    result = verify(
        [att], target="café", nodes=nodes,
        trusted={"签署者α": pub_b64(KEY_1)},
    )
    assert result.trusted is True


# --------------------------------------------------------- failure modes


def test_missing_attestation_for_target_and_dependency():
    result = verify([])
    assert result.trusted is False
    assert result.reasons == {
        "core": ["missing_attestation"],
        "util": ["missing_attestation"],
    }


def test_missing_dependency_attestation_is_digest_mismatch():
    util_only = [make_attestation("util", F_UTIL, {"core": GOLDEN_CORE_DIGEST})]
    result = verify(util_only)
    assert result.trusted is False
    assert result.reasons == {
        "core": ["missing_attestation"],
        # The mapped digest cannot match an attestation that does not exist.
        "util": ["dependency_digest_mismatch"],
    }


def test_fingerprint_mismatch_is_reported_per_node_and_propagates_digest_check():
    # util claims a stale fingerprint but re-signs; core stays valid.
    core_att = make_attestation("core", F_CORE, {})
    util_att = make_attestation(
        "util", "deadbeef" * 8, {"core": attestation_digest(core_att)}
    )
    result = verify([core_att, util_att])
    assert result.trusted is False
    assert result.reasons == {"util": ["fingerprint_mismatch"]}


def test_unknown_signer():
    att = make_attestation(
        "core", F_CORE, {}, signer=UNTRUSTED, signing_key=KEY_3
    )
    result = verify([att], target="core")
    assert result.trusted is False
    assert result.reasons == {"core": ["unknown_signer"]}


def test_unknown_signer_does_not_attempt_signature_check():
    # Signed by KEY_3 and attributed to a signer id that is absent from the
    # trusted table: only unknown_signer, never invalid_signature.
    att = make_attestation(
        "core", F_CORE, {}, signer=UNTRUSTED, signing_key=KEY_3
    )
    result = verify([att], target="core", trusted={SIGNER_1: pub_b64(KEY_1)})
    assert result.reasons == {"core": ["unknown_signer"]}


def test_wrong_key_under_known_signer_is_invalid_signature():
    # A known signer id whose trusted key differs from the signing key.
    att = make_attestation(
        "core", F_CORE, {}, signer=SIGNER_1, signing_key=KEY_3
    )
    result = verify([att], target="core")
    assert result.reasons == {"core": ["invalid_signature"]}


def test_invalid_signature():
    good = make_attestation("core", F_CORE, {})
    tampered = good.model_copy(
        update={"signature": b64(b"\x00" * 64)}
    )
    result = verify([tampered], target="core")
    assert result.trusted is False
    assert result.reasons == {"core": ["invalid_signature"]}


def test_signature_verifies_against_exact_canonical_bytes():
    att = make_attestation("core", F_CORE, {})
    # A signature over the hex digest alone must NOT verify.
    wrong_sig = b64(KEY_1.sign(bytes.fromhex(attestation_digest(att))))
    wrong = att.model_copy(update={"signature": wrong_sig})
    with pytest.raises(InvalidSignature):
        Ed25519PublicKey.from_public_bytes(
            base64.b64decode(TRUSTED_KEYS[SIGNER_1])
        ).verify(base64.b64decode(wrong_sig), attestation_message(att))
    result = verify([wrong], target="core")
    assert result.reasons == {"core": ["invalid_signature"]}


def test_dependency_missing_from_map():
    core_att = make_attestation("core", F_CORE, {})
    # util maps no dependencies at all.
    util_att = make_attestation("util", F_UTIL, {})
    result = verify([core_att, util_att])
    assert result.reasons == {"util": ["dependency_mismatch"]}


def test_unexpected_dependency_in_map():
    # docs exists in the graph but is not a direct dependency of core.
    docs_att = make_attestation("docs", graph_fingerprint("docs-src-v1", []), {})
    docs_digest = attestation_digest(docs_att)
    bad_core = make_attestation("core", F_CORE, {"docs": docs_digest})
    result = verify([bad_core], target="core")
    assert result.reasons == {"core": ["dependency_mismatch"]}


def test_dependency_digest_mismatch():
    core_att = make_attestation("core", F_CORE, {})
    # util maps an arbitrary but well-formed digest for core.
    util_att = make_attestation("util", F_UTIL, {"core": "00" * 32})
    result = verify([core_att, util_att])
    assert result.reasons == {"util": ["dependency_digest_mismatch"]}


def test_tampered_dependency_fingerprint_breaks_parent_chain():
    # core attests a stale fingerprint (re-signed); util honestly claims the
    # true graph fingerprint and links to the tampered core digest, so the
    # digest chain itself is consistent -- only core's fingerprint fails.
    claimed_core_fp = "11" * 32
    core_att = make_attestation("core", claimed_core_fp, {})
    honest_util = make_attestation(
        "util", F_UTIL, {"core": attestation_digest(core_att)}
    )
    result = verify([core_att, honest_util])
    assert result.trusted is False
    assert result.reasons == {"core": ["fingerprint_mismatch"]}

    # If util instead links to a digest that no core attestation actually has,
    # the broken link is reported as dependency_digest_mismatch on util.
    stale_util = make_attestation("util", F_UTIL, {"core": "22" * 32})
    result = verify([core_att, stale_util])
    assert result.reasons == {
        "core": ["fingerprint_mismatch"],
        "util": ["dependency_digest_mismatch"],
    }


def test_multiple_reasons_collected_and_sorted():
    # Signed with KEY_3 while trusted under SIGNER_1 -> invalid_signature;
    # wrong fingerprint and an extra dependency add two more reasons.
    att = make_attestation(
        "core",
        "99" * 32,
        {"docs": attestation_digest(make_attestation("docs", "x", {}))},
        signer=SIGNER_1,
        signing_key=KEY_3,
    )
    result = verify([att], target="core")
    assert result.trusted is False
    assert result.reasons == {
        "core": [
            "dependency_mismatch",
            "fingerprint_mismatch",
            "invalid_signature",
        ]
    }


def test_reasons_sorted_by_node_name():
    nodes = [
        node("z", "z"),
        node("m", "m"),
        node("a", "a", ["z", "m"]),
    ]
    result = verify_provenance(nodes, "a", [], {})
    assert list(result.reasons) == ["a", "m", "z"]


def test_diamond_closure_counts_shared_dependency_once():
    # t -> a -> s, t -> b -> s: s is in the closure once and needs one proof.
    nodes = [
        node("s", "s"),
        node("a", "a", ["s"]),
        node("b", "b", ["s"]),
        node("t", "t", ["a", "b"]),
    ]
    f_s = graph_fingerprint("s", [])
    f_a = graph_fingerprint("a", [("s", f_s)])
    f_b = graph_fingerprint("b", [("s", f_s)])
    f_t = graph_fingerprint("t", [("a", f_a), ("b", f_b)])
    att_s = make_attestation("s", f_s, {})
    att_a = make_attestation("a", f_a, {"s": attestation_digest(att_s)})
    att_b = make_attestation("b", f_b, {"s": attestation_digest(att_s)})
    att_t = make_attestation(
        "t", f_t,
        {"a": attestation_digest(att_a), "b": attestation_digest(att_b)},
    )
    result = verify_provenance(nodes, "t", [att_t, att_a, att_b, att_s], dict(TRUSTED_KEYS))
    assert result.trusted is True

    # Removing the shared proof flags s once; a/b honestly linked to its digest
    # so their references dangle, while t links to the intact a/b digests.
    result = verify_provenance(nodes, "t", [att_t, att_a, att_b], dict(TRUSTED_KEYS))
    assert result.reasons == {
        "a": ["dependency_digest_mismatch"],
        "b": ["dependency_digest_mismatch"],
        "s": ["missing_attestation"],
    }


def test_missing_proofs_for_whole_closure_list_every_node_once():
    result = verify_provenance(chain_nodes(), "util", [], dict(TRUSTED_KEYS))
    assert result.reasons == {
        "core": ["missing_attestation"],
        "util": ["missing_attestation"],
    }


def test_empty_trusted_keys_marks_every_signer_unknown():
    att = make_attestation("core", F_CORE, {})
    result = verify([att], target="core", trusted={})
    assert result.reasons == {"core": ["unknown_signer"]}


def test_extra_trusted_keys_are_ignored():
    trusted = dict(TRUSTED_KEYS)
    trusted["nobody-ever-used"] = pub_b64(KEY_3)
    result = verify(signed_chain(), trusted=trusted)
    assert result.trusted is True


# ----------------------------------------------------------- hard errors


def test_duplicate_node_graph_error():
    with pytest.raises(PlanError) as exc:
        verify_provenance(
            [node("core", "x"), node("core", "y")],
            "core",
            [],
            dict(TRUSTED_KEYS),
        )
    assert exc.value.code == "duplicate_node"


def test_missing_dependency_graph_error():
    with pytest.raises(PlanError) as exc:
        verify_provenance(
            [node("a", "a", ["ghost"])], "a", [], dict(TRUSTED_KEYS)
        )
    assert exc.value.code == "missing_dependency"


def test_cycle_graph_error():
    with pytest.raises(PlanError) as exc:
        verify_provenance(
            [node("a", "a", ["b"]), node("b", "b", ["a"])],
            "a",
            [],
            dict(TRUSTED_KEYS),
        )
    assert exc.value.code == "cycle_detected"


def test_unknown_target():
    with pytest.raises(PlanError) as exc:
        verify(signed_chain(), target="ghost")
    assert exc.value.code == "unknown_target"
    assert exc.value.details == {"target": "ghost"}


def test_duplicate_attestation():
    with pytest.raises(PlanError) as exc:
        verify([make_attestation("core", F_CORE, {}),
                make_attestation("core", F_CORE, {})], target="core")
    assert exc.value.code == "duplicate_attestation"
    assert exc.value.details == {"nodes": ["core"]}


def test_attestation_outside_closure():
    # docs is in the graph but not reachable from util.
    docs_att = make_attestation("docs", graph_fingerprint("docs-src-v1", []), {})
    with pytest.raises(PlanError) as exc:
        verify(signed_chain() + [docs_att])
    assert exc.value.code == "attestation_outside_closure"
    assert exc.value.details == {"nodes": ["docs"], "target": "util"}


def test_unknown_attestation_node_not_in_graph():
    att = make_attestation("phantom", F_CORE, {})
    with pytest.raises(PlanError) as exc:
        verify([att])
    assert exc.value.code == "unknown_attestation_node"
    assert exc.value.details == {"names": ["phantom"]}


def test_malformed_signature_base64():
    good = make_attestation("core", F_CORE, {})
    bad = good.model_copy(update={"signature": "not!base64!!"})
    with pytest.raises(PlanError) as exc:
        verify([bad], target="core")
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"signatures": ["core"]}


def test_malformed_signature_length():
    good = make_attestation("core", F_CORE, {})
    bad = good.model_copy(update={"signature": b64(b"\x00" * 32)})
    with pytest.raises(PlanError) as exc:
        verify([bad], target="core")
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"signatures": ["core"]}


def test_urlsafe_base64_signature_is_malformed():
    raw = bytes(range(64))
    sig = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    good = make_attestation("core", F_CORE, {})
    bad = good.model_copy(update={"signature": sig})
    with pytest.raises(PlanError) as exc:
        verify([bad], target="core")
    assert exc.value.code == "malformed_encoding"


def test_malformed_trusted_key_base64():
    with pytest.raises(PlanError) as exc:
        verify(signed_chain(), trusted={SIGNER_1: "***"})
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"trusted_keys": [SIGNER_1]}


def test_malformed_trusted_key_length():
    with pytest.raises(PlanError) as exc:
        verify(signed_chain(), trusted={SIGNER_1: b64(b"\x01" * 31)})
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"trusted_keys": [SIGNER_1]}


def test_unused_malformed_trusted_key_is_still_a_hard_error():
    trusted = dict(TRUSTED_KEYS)
    trusted["unused"] = "not-base64"
    with pytest.raises(PlanError) as exc:
        verify(signed_chain(), trusted=trusted)
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"trusted_keys": ["unused"]}


# --------------------------------------------------------------- HTTP


def _request_payload(attestations: list[Attestation], **overrides):
    payload = {
        "nodes": [
            {
                "name": n.name,
                "content_digest": n.content_digest,
                "dependency_names": list(n.dependency_names),
            }
            for n in chain_nodes()
        ],
        "target": "util",
        "attestations": [
            {
                "node": a.node,
                "fingerprint": a.fingerprint,
                "signer": a.signer,
                "dependencies": dict(a.dependencies),
                "signature": a.signature,
            }
            for a in attestations
        ],
        "trusted_keys": dict(TRUSTED_KEYS),
    }
    payload.update(overrides)
    return payload


def test_http_valid_chain():
    response = client.post(
        "/provenance/verify", json=_request_payload(signed_chain())
    )
    assert response.status_code == 200
    assert response.json() == {"trusted": True, "reasons": {}}


def test_http_failure_verdict_is_200_with_sorted_reasons():
    response = client.post("/provenance/verify", json=_request_payload([]))
    assert response.status_code == 200
    body = response.json()
    assert body["trusted"] is False
    assert list(body["reasons"]) == ["core", "util"]


def test_http_unknown_target_is_400():
    response = client.post(
        "/provenance/verify", json=_request_payload([], target="ghost")
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unknown_target"


def test_http_duplicate_attestation_is_400():
    dupes = signed_chain() + [make_attestation("core", F_CORE, {})]
    response = client.post(
        "/provenance/verify", json=_request_payload(dupes)
    )
    assert response.status_code == 400
    assert response.json()["code"] == "duplicate_attestation"


def test_http_malformed_encoding_is_400():
    bad = make_attestation("core", F_CORE, {})
    payload = _request_payload(
        [bad], target="core",
        trusted_keys={SIGNER_1: "%%%"},
    )
    response = client.post("/provenance/verify", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "malformed_encoding"


def test_http_graph_cycle_is_400():
    payload = _request_payload([])
    payload["nodes"] = [
        {"name": "a", "content_digest": "a", "dependency_names": ["b"]},
        {"name": "b", "content_digest": "b", "dependency_names": ["a"]},
    ]
    payload["target"] = "a"
    response = client.post("/provenance/verify", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "cycle_detected"


def test_http_unknown_field_is_422():
    payload = _request_payload(signed_chain())
    payload["bogus"] = 1
    response = client.post("/provenance/verify", json=payload)
    assert response.status_code == 422


def test_http_empty_signature_is_422():
    atts = signed_chain()
    payload = _request_payload(atts)
    payload["attestations"][0]["signature"] = ""
    response = client.post("/provenance/verify", json=payload)
    assert response.status_code == 422
