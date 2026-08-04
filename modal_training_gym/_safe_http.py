"""Small stdlib-only HTTP helpers that strip auth headers on cross-origin redirects.

Used by fire-and-forget status reporters inside training containers, where adding
``httpx``/third-party clients is undesirable.  ``urllib.request`` forwards custom
headers on redirects by default, so a compromised dashboard host (or any redirect
to an attacker-controlled host) would leak credentials.  The helpers below
remove sensitive headers before following any redirect whose scheme or host
differs from the original request, and refuse to follow cross-origin redirects
entirely.
"""

from __future__ import annotations

from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _CrossOriginSafeRedirectHandler(HTTPRedirectHandler):
    """Follow redirects, but never forward sensitive auth headers to another origin."""

    _SENSITIVE_HEADERS = frozenset(
        {
            "authorization",
            "modal-key",
            "modal-secret",
            "proxy-authorization",
        }
    )

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        m = req.get_method()
        # Follow the same redirects urllib follows by default, plus preserve-method
        # 307/308 redirects for POST.  Refuse everything else.
        if not (
            (code in (301, 302, 303, 307, 308) and m in ("GET", "HEAD"))
            or (code in (301, 302, 303, 307, 308) and m == "POST")
        ):
            raise HTTPError(req.full_url, code, msg, headers, fp)

        # 301/302/303 historically collapse POST -> GET and drop the body.
        new_method = m
        new_data = req.data
        if code in (301, 302, 303) and m == "POST":
            new_method = "GET"
            new_data = None

        # Drop content headers when the body is dropped; preserve Content-Type
        # when the body is preserved (Content-Length is recomputed by Request).
        drop_headers = {"content-length"}
        if new_data is None:
            drop_headers.add("content-type")
        newheaders = {
            k: v for k, v in req.headers.items() if k.lower() not in drop_headers
        }

        old = urlsplit(req.full_url)
        new = urlsplit(newurl)
        same_origin = (
            old.scheme.lower(),
            old.netloc.lower(),
        ) == (
            new.scheme.lower(),
            new.netloc.lower(),
        )

        new_req = Request(
            newurl,
            data=new_data,
            headers=newheaders,
            origin_req_host=req.origin_req_host,
            unverifiable=True,
            method=new_method,
        )

        if not same_origin:
            for name in self._SENSITIVE_HEADERS:
                if new_req.has_header(name):
                    new_req.remove_header(name)
            # Refuse to follow cross-origin redirects. The destination is not the
            # configured dashboard and should not receive the request body.
            raise URLError(
                f"refusing cross-origin redirect from {old.netloc} to {new.netloc}"
            )

        return new_req


def origin(url: str) -> str | None:
    """Return the ``scheme://netloc`` origin of ``url`` in lowercase, or ``None``."""
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    return None


def post(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float | None = None,
) -> Any:
    """POST ``body`` to ``url`` and return the response object.

    Redirects are followed, but ``Authorization``, ``Modal-Key``,
    ``Modal-Secret``, and ``Proxy-Authorization`` headers are removed before
    following any redirect whose scheme or host differs from the original URL,
    and cross-origin redirects are blocked entirely.
    """
    request = Request(url, data=body, headers=headers, method="POST")
    opener = build_opener(_CrossOriginSafeRedirectHandler)
    return opener.open(request, timeout=timeout)
