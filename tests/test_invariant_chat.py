import pytest
from urllib.parse import urlparse


# Adversarial SSRF payloads targeting internal/restricted resources
SSRF_PAYLOADS = [
    # Cloud metadata endpoints
    "http://169.254.169.254/latest/meta-data/",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://169.254.169.254/latest/user-data",
    "http://169.254.169.254/computeMetadata/v1/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
    "http://100.100.100.200/latest/meta-data/",  # Alibaba Cloud metadata
    # IPv6 metadata
    "http://[fd00:ec2::254]/latest/meta-data/",
    # Internal network ranges
    "http://10.0.0.1/admin",
    "http://10.255.255.255/secret",
    "http://172.16.0.1/internal",
    "http://172.31.255.255/config",
    "http://192.168.0.1/router",
    "http://192.168.1.1/admin",
    "http://127.0.0.1/etc/passwd",
    "http://localhost/admin",
    "http://localhost:8080/actuator/env",
    "http://localhost:9200/_cat/indices",  # Elasticsearch
    "http://localhost:6379/",  # Redis
    "http://localhost:5432/",  # PostgreSQL
    "http://localhost:27017/",  # MongoDB
    # DNS rebinding / bypass attempts
    "http://0.0.0.0/secret",
    "http://0177.0.0.1/etc/passwd",  # Octal encoding
    "http://2130706433/",  # Decimal encoding of 127.0.0.1
    "http://0x7f000001/",  # Hex encoding of 127.0.0.1
    # Protocol smuggling
    "file:///etc/passwd",
    "file:///etc/shadow",
    "file:///proc/self/environ",
    "file:///var/run/secrets/kubernetes.io/serviceaccount/token",
    "dict://localhost:11211/stat",  # Memcached
    "gopher://localhost:6379/_FLUSHALL",  # Redis via gopher
    "ftp://localhost/etc/passwd",
    # URL with credentials to bypass naive checks
    "http://attacker.com@169.254.169.254/latest/meta-data/",
    "http://169.254.169.254#@example.com/latest/meta-data/",
    # Kubernetes internal services
    "http://kubernetes.default.svc/api/v1/secrets",
    "http://10.96.0.1/api/v1/namespaces/default/secrets",
    # Internal hostnames
    "http://internal-service/api/keys",
    "http://db.internal/",
    "http://redis.internal:6379/",
]


def is_safe_url(url: str) -> bool:
    """
    Security invariant: URLs must be validated before fetching.
    Only allow HTTP/HTTPS to non-private, non-loopback, non-metadata hosts.
    """
    from fastapi import HTTPException
    from llamafactory.api.common import check_ssrf_url

    try:
        check_ssrf_url(url)
        return True
    except HTTPException:
        return False


@pytest.mark.parametrize("payload", SSRF_PAYLOADS)
def test_ssrf_url_validation_blocks_adversarial_inputs(payload):
    """
    Invariant: The chat API must never fetch resources from user-supplied URLs
    that point to internal network addresses, cloud metadata endpoints, loopback
    addresses, or non-HTTP(S) schemes. Any URL validation function used before
    requests.get() must reject all known SSRF attack vectors.
    """
    result = is_safe_url(payload)
    assert result is False, (
        f"SECURITY VIOLATION: URL '{payload}' was not blocked by URL validation. "
        f"This URL could be used for SSRF attacks to access internal resources. "
        f"The chat API must validate all user-supplied media URLs before fetching them."
    )


@pytest.mark.parametrize("payload", SSRF_PAYLOADS)
def test_ssrf_url_scheme_restriction(payload):
    """
    Invariant: Only http and https schemes should be permitted for external
    resource fetching. Any other scheme (file, gopher, dict, ftp, etc.) must
    be rejected unconditionally.
    """
    parsed = urlparse(payload)
    if parsed.scheme not in ("http", "https"):
        result = is_safe_url(payload)
        assert result is False, (
            f"SECURITY VIOLATION: Non-HTTP(S) scheme '{parsed.scheme}' in URL '{payload}' "
            f"was not blocked. Only http/https schemes should be allowed."
        )


def test_safe_url_is_allowed():
    """
    Sanity check: Legitimate external URLs should pass validation.
    This ensures the security check doesn't block all traffic.
    """
    safe_urls = [
        "https://example.com/image.jpg",
        "http://cdn.example.org/video.mp4",
        "https://storage.googleapis.com/bucket/audio.wav",
        "https://s3.amazonaws.com/bucket/file.png",
    ]
    for url in safe_urls:
        assert is_safe_url(url) is True, (
            f"Legitimate URL '{url}' was incorrectly blocked by URL validation."
        )


def test_empty_and_malformed_urls_are_rejected():
    """
    Invariant: Empty, None-like, or malformed URLs must never be fetched.
    """
    malformed = [
        "",
        "not-a-url",
        "://missing-scheme",
        "http://",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
    ]
    for url in malformed:
        result = is_safe_url(url)
        assert result is False, (
            f"Malformed/dangerous URL '{url}' was not rejected by URL validation."
        )