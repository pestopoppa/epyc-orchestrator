#!/usr/bin/env python3
"""Model-free isolated CRM/Slack control; caller must own a CPU claim."""
import argparse
import os
import sys
from pathlib import Path

from harness.crm_provision import PrivateServices
from harness.crm_receipt import Receipt
from harness.crm_services import NativeServices, load_pinned_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpus", "seed", "crm-source", "slack-source", "receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    receipt = Receipt(Path(args.receipt), {"scope": "model-free-native-services-control", "cpus": args.cpus})
    private = PrivateServices(cpus=args.cpus, seed_path=Path(args.seed), receipt=receipt)
    try:
        private.start()
        os.environ.update(SALESFORCE_BASE_URL=private.crm_url + "/", SALESFORCE_GRANT_TYPE="client_credentials",
                          SALESFORCE_CLIENT_ID="17817554-086b-83d1-bf9d-69026221f529",
                          SALESFORCE_CLIENT_SECRET="mcp-secret-123", SLACK_API_URL=private.slack_url,
                          USER_ACCESS_TOKEN="slack_token_sales_ops")
        crm = load_pinned_module(Path(args.crm_source), "crm")
        slack = load_pinned_module(Path(args.slack_source), "slack")
        services = NativeServices(crm, slack, reset=private.reset, teardown=private.close)
        initial = services.reset()
        receipt.append("initial", initial)
        note = services.invoke("create_record", {"module_name": "Notes", "attributes": {
            "name": "Daily Activity Summary - Meridian Group", "description": "customer engagement tracked and logged"}})
        receipt.append("note", note)
        message = services.invoke("post_message", {"channel_name": "sales", "text": "Meridian Group activity summary"})
        receipt.append("message", message)
        state = services.snapshot()
        receipt.append("state", state)
        if not any(x.get("attributes", {}).get("name") == "Daily Activity Summary - Meridian Group"
                   for x in state["salesforce"]["Notes"]):
            raise RuntimeError("created note missing from actual service snapshot")
        if not any("Meridian Group activity summary" in x.get("text", "")
                   for x in state["slack"]["channels"]["sales"]):
            raise RuntimeError("sent message missing from actual service snapshot")
        reset = services.reset()
        receipt.append("reset", reset)
        if reset != initial:
            raise RuntimeError("second reset does not reproduce initial judge state")
        receipt.append("control_result", {"pass": True, "model_invoked": False})
        print("MODEL_FREE_SERVICE_CONTROL_PASS")
        return 0
    except BaseException as exc:
        receipt.append("control_failure", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        try:
            private.close()
            receipt.append("cleanup", {"ok": True})
        except BaseException as exc:
            receipt.append("cleanup", {"ok": False, "error": str(exc)})
            raise
        finally:
            receipt.close()


if __name__ == "__main__":
    sys.exit(main())
