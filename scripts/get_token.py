#!/usr/bin/env python3
"""
Customer OAuth PKCE token helper for local development.

Simulates the mobile app's Zitadel PKCE login flow from the command line.

Two modes:
  --auto   (default) Starts a local HTTP server to capture the redirect.
           Requires the redirect URI to be registered in Zitadel.
  --manual Opens browser, then asks you to paste the redirect URL manually.
           Works even if redirect URI is not registered (e.g. ceramicraft://login).

Usage:
    uv run python scripts/get_token.py
    uv run python scripts/get_token.py --manual
    uv run python scripts/get_token.py --user-ms-url http://localhost:8083
"""

import argparse
import base64
import hashlib
import http.server
import json
import secrets
import sys
import threading
import urllib.parse
import webbrowser

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: uv pip install httpx")
    sys.exit(1)

# Zitadel configuration (same as mobile app login.tsx)
ZITADEL_HOST = "https://cerami-t6ihrd.us1.zitadel.cloud"
CLIENT_ID = "361761429302373082"
SCOPES = "openid profile email offline_access urn:zitadel:iam:user:metadata custom:local_userid"

# For auto mode
AUTO_REDIRECT_PORT = 18899
AUTO_REDIRECT_URI = f"http://localhost:{AUTO_REDIRECT_PORT}/callback"

# For manual mode — use the same redirect URI as the mobile app
MANUAL_REDIRECT_URI = "ceramicraft://login"


def _generate_code_verifier() -> str:
    return secrets.token_urlsafe(64)[:128]


def _generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            _CallbackHandler.code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h2>Login successful! You can close this tab.</h2>")
        else:
            error = params.get("error", ["unknown"])[0]
            desc = params.get("error_description", [""])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"<h2>Error: {error}</h2><p>{desc}</p>".encode())

    def log_message(self, *args) -> None:  # noqa: ANN002
        pass


def _wait_for_code_auto(timeout: int = 120) -> str:
    server = http.server.HTTPServer(("127.0.0.1", AUTO_REDIRECT_PORT), _CallbackHandler)
    server.timeout = timeout

    def _serve() -> None:
        while _CallbackHandler.code is None:
            server.handle_request()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    server.server_close()

    if _CallbackHandler.code is None:
        print("ERROR: Timed out waiting for authorization code.")
        sys.exit(1)
    return _CallbackHandler.code


def _wait_for_code_manual() -> str:
    print()
    print(
        "After login, the browser will try to redirect to ceramicraft://login?code=...&state=..."
    )
    print("This will FAIL (no app installed). That's expected!")
    print()
    print("Copy the FULL URL from the browser address bar and paste it below.")
    print("(It will look like: ceramicraft://login?code=XXXX&state=YYYY)")
    print()
    raw = input("Paste redirect URL here: ").strip()
    parsed = urllib.parse.urlparse(raw)
    params = urllib.parse.parse_qs(parsed.query)
    if "code" not in params:
        print(f"ERROR: No 'code' parameter found in URL: {raw}")
        sys.exit(1)
    return params["code"][0]


def _exchange_code(code: str, redirect_uri: str, code_verifier: str) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{ZITADEL_HOST}/oauth/v2/token",
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
        )
        if resp.status_code != 200:
            print(f"ERROR: Token exchange failed ({resp.status_code}):")
            print(resp.text)
            sys.exit(1)
        return resp.json()


def _register_user(user_ms_url: str, access_token: str) -> None:
    print(f"Registering user via {user_ms_url}/user-ms/v1/customer/oauth-callback ...")
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{user_ms_url}/user-ms/v1/customer/oauth-callback",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if resp.status_code == 200:
            print(f"  Result: {resp.json()}")
        else:
            print(f"  WARNING: oauth-callback returned {resp.status_code}: {resp.text}")
            print("  Continuing anyway (user may already exist)...")


def _refresh_token(refresh_token: str) -> dict | None:
    print("Refreshing token to embed local_userid metadata...")
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{ZITADEL_HOST}/oauth/v2/token",
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
        if resp.status_code == 200:
            print("  Token refreshed with local_userid metadata!")
            return resp.json()
        print(f"  WARNING: Refresh failed ({resp.status_code}), using initial token")
        return None


def _decode_token(token: str) -> None:
    try:
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        print("\nToken claims:")
        print(f"  sub:    {claims.get('sub')}")
        print(f"  email:  {claims.get('email')}")
        print(f"  name:   {claims.get('name', 'N/A')}")
        metadata = claims.get("urn:zitadel:iam:user:metadata", {})
        if "local_userid" in metadata:
            local_id = base64.b64decode(metadata["local_userid"]).decode()
            print(f"  local_userid: {local_id}  <-- this is the internal user ID")
        else:
            print("  local_userid: NOT FOUND (first-time user, see step below)")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get customer OAuth token for CS Agent"
    )
    parser.add_argument(
        "--user-ms-url",
        default="http://localhost:8083",
        help="Base URL of user-ms (default: http://localhost:8083)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual mode: paste redirect URL instead of local server",
    )
    args = parser.parse_args()

    manual = args.manual
    redirect_uri = MANUAL_REDIRECT_URI if manual else AUTO_REDIRECT_URI

    # Step 1: PKCE parameters
    code_verifier = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)
    state = secrets.token_urlsafe(16)

    # Step 2: Build authorization URL
    auth_params = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPES,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    auth_url = f"{ZITADEL_HOST}/oauth/v2/authorize?{auth_params}"

    print("=" * 60)
    print("CeramiCraft Customer Token Helper (PKCE)")
    print("=" * 60)
    print(f"Mode: {'manual' if manual else 'auto'}")
    print(f"Redirect URI: {redirect_uri}")
    print()
    print("Opening browser for Zitadel login...")
    if manual:
        print(f"\nIf browser doesn't open, visit:\n  {auth_url}")
    webbrowser.open(auth_url)

    # Step 3: Get authorization code
    if manual:
        code = _wait_for_code_manual()
    else:
        print("Waiting for authorization code on local server...")
        code = _wait_for_code_auto()

    print("Authorization code received!\n")

    # Step 4: Exchange code for tokens
    print("Exchanging code for tokens...")
    tokens = _exchange_code(code, redirect_uri, code_verifier)
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")
    id_token = tokens.get("id_token", "")
    print("  Tokens received!")

    # Step 5: Register user via oauth-callback
    _register_user(args.user_ms_url, access_token)

    # Step 6: Refresh to get token with local_userid
    final_id_token = id_token
    if refresh_token:
        new_tokens = _refresh_token(refresh_token)
        if new_tokens:
            final_id_token = new_tokens.get("id_token", id_token)
    else:
        print("WARNING: No refresh_token. Token may lack local_userid metadata.")
        print(
            "  Check that Zitadel app has 'offline_access' scope and refresh token grant enabled."
        )

    # Output
    print("\n" + "=" * 60)
    print("FINAL ID TOKEN (use this with cs-agent):")
    print("=" * 60)
    print(final_id_token)
    print("=" * 60)
    _decode_token(final_id_token)

    print("\n--- Copy-paste commands ---")
    print("\nPowerShell:")
    print(f'  $env:TOKEN = "{final_id_token}"')
    print("\nBash:")
    print(f'  export TOKEN="{final_id_token}"')


if __name__ == "__main__":
    main()
