#!/usr/bin/env python3
"""
Interactive CLI chat client for CeramiCraft Customer Support Agent.

Starts an interactive session: sends messages to the CS Agent REST API
and prints replies in a loop until Ctrl+C.

Usage:
    uv run python scripts/chat.py
    uv run python scripts/chat.py --base-url http://localhost:8080
"""

import argparse
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CS Agent chat client")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="CS Agent base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    # Require token
    token = input("Token (required): ").strip()
    if not token:
        print("ERROR: Token is required.")
        sys.exit(1)

    # Optional thread_id
    thread_id = input("Thread ID (press Enter to create new): ").strip() or None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    print()
    print("=" * 50)
    print("CeramiCraft CS Agent Chat")
    print(f"Server: {args.base_url}")
    print(f"Thread: {thread_id or '(new)'}")
    print("Type your message and press Enter. Ctrl+C to quit.")
    print("=" * 50)
    print()

    with httpx.Client(timeout=args.timeout) as client:
        while True:
            try:
                message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not message:
                continue

            payload: dict = {"message": message}
            if thread_id:
                payload["thread_id"] = thread_id

            try:
                resp = client.post(
                    f"{args.base_url}/chat",
                    headers=headers,
                    json=payload,
                )
            except httpx.ConnectError:
                print(f"  [ERROR] Cannot connect to {args.base_url}")
                continue
            except httpx.ReadTimeout:
                print("  [ERROR] Request timed out")
                continue

            if resp.status_code != 200:
                print(f"  [ERROR] HTTP {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            reply = data.get("reply", "(no reply)")
            returned_thread = data.get("thread_id", "")

            # Capture thread_id from first response
            if not thread_id and returned_thread:
                thread_id = returned_thread
                print(f"  [Thread: {thread_id}]")

            print(f"Agent: {reply}")
            print()


if __name__ == "__main__":
    main()
