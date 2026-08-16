#!/usr/bin/env python3
"""Reconcile actual TPU Queued Resources against the desired TRC plan.

Meant to be triggered periodically by cron/systemd -- this script does one
pass and exits, it does not loop or sleep itself. Most of the TRC grant is
spot capacity, which gets preempted; when that happens the slice needs its
stale queued resource deleted and a fresh one created. Each run:

  1. Computes the desired list of slices from trc_quota.py (same split
     logic plan_tpu_requests.py uses).
  2. Queries the actual queued-resource state per zone via gcloud.
  3. Missing slices get created (spot and on-demand alike).
  4. Spot slices present but not in a healthy state get *deleted only* --
     the recreate is left to the next run, once gcloud actually reports
     them gone. Deleting and recreating in the same pass risks racing
     gcloud's own eventual consistency (delete may not be immediate), so
     recovery for an unhealthy slice takes up to two cron cycles by
     design, not by accident.
  5. On-demand slices present but not in a healthy state are *never*
     auto-deleted -- on-demand capacity doesn't get preempted the way spot
     does, so an unhealthy on-demand slice usually means something is
     actually wrong (not just "waiting for capacity again"), and it's a
     real paid resource that shouldn't be torn down without a human
     looking at it first. These get flagged for manual review instead.

Usage (e.g. cron every 5 minutes):

    python reconcile.py            # act (delete stale / create missing)
    python reconcile.py --dry-run  # only print what it would do

HEALTHY_STATES below is a best guess, not verified against a live account
(no gcloud available while writing this). Before trusting this in cron,
run once by hand and compare against real output:
`gcloud compute tpus queued-resources list --zone=<zone> --format="value(name,state)"`
-- adjust HEALTHY_STATES if the state strings differ.
"""

import argparse
import subprocess
import sys

from plan_tpu_requests import build_command, build_delete_command, build_list_command, render_command, slice_name, split_entry
from trc_quota import PROJECT_ID, QUOTA

HEALTHY_STATES = {"ACTIVE", "PROVISIONING", "WAITING_FOR_RESOURCES", "CREATING"}


def desired_slices() -> list[dict]:
    """One dict per slice the current trc_quota.py wants to exist."""
    slices = []
    for entry in QUOTA:
        for i, req in enumerate(split_entry(entry)):
            slices.append({"name": slice_name(req, i), "zone": req["zone"], "entry": req, "index": i})
    return slices


def actual_states(zone: str) -> dict:
    """{name: state} for queued resources gcloud currently reports in `zone`."""
    result = subprocess.run(
        build_list_command(zone),
        capture_output=True, text=True, check=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        name, state = parts[0], parts[-1]
        states[name] = state
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print actions without deleting/creating anything")
    args = parser.parse_args()

    if not PROJECT_ID:
        sys.exit("Set the TPU_PROJECT_ID environment variable before running this script, e.g.:\n" "  export TPU_PROJECT_ID=my-gcp-project")

    slices = desired_slices()
    zones = sorted({s["zone"] for s in slices})
    zone_states = {zone: actual_states(zone) for zone in zones}

    to_create, to_delete, needs_review = [], [], []
    for s in slices:
        state = zone_states[s["zone"]].get(s["name"])
        tier = s["entry"]["tier"]
        if state is None:
            print(f"MISSING   {s['name']} ({s['zone']}) -- will create")
            to_create.append(s)
        elif state not in HEALTHY_STATES:
            if tier == "on-demand":
                print(f"UNHEALTHY {s['name']} ({s['zone']}) state={state} -- on-demand, NOT auto-deleting; needs manual review")
                needs_review.append(s)
            else:
                print(f"UNHEALTHY {s['name']} ({s['zone']}) state={state} -- will delete (recreated next run)")
                to_delete.append(s)
        else:
            print(f"OK        {s['name']} ({s['zone']}) state={state}")

    if not to_create and not to_delete and not needs_review:
        print("\nAll slices healthy, nothing to do.")
        return

    review_note = f", {len(needs_review)} on-demand need manual review" if needs_review else ""
    print(f"\n{len(to_delete)} to delete, {len(to_create)} to create{review_note}.")

    for s in to_delete:
        cmd = build_delete_command(s["name"], s["zone"])
        print(render_command(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    for s in to_create:
        cmd = build_command(s["entry"], s["index"])
        print(render_command(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
