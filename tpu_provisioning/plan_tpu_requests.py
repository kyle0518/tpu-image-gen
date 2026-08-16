#!/usr/bin/env python3
"""Turn a TRC quota grant into `gcloud ... queued-resources` commands.

TRC grants a fixed chip count per (zone, TPU generation, tier), but asking
for that whole count in a single Queued Resource on spot capacity doesn't
reliably get fulfilled -- e.g. a single v5e-64 or v6e-64 spot QR has failed
in practice for this project, while the same total chips split into several
smaller QRs (v5e-32 x2, v6e-16 x4) has worked. This script encodes that
split (see MAX_CHIPS_PER_SPOT_REQUEST in trc_quota.py) so the gcloud
commands don't have to be reassembled by hand every time the grant renews.

Export TPU_PROJECT_ID (and optionally TPU_GCLOUD_ACCOUNT -- see
trc_quota.py) to match your current TRC grant, then:

    python plan_tpu_requests.py            # print the create commands
    python plan_tpu_requests.py --run      # print AND execute them
    python plan_tpu_requests.py --run --yes  # execute without confirmation

Every run also writes a full command reference -- check (list), create,
describe, and delete, each section its own commented ```bash``` block -- to
tpu_commands.md (see --output) so cleanup and inspection commands don't have
to be hand-assembled later, and a section can be copied straight from a web
view (e.g. GitHub) without opening a terminal.

Verify accelerator-type / runtime-version strings for your zones before
relying on this (they can change): `gcloud compute tpus accelerator-types
list --zone=<zone>` and `gcloud compute tpus versions list --zone=<zone>`.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from trc_quota import MAX_CHIPS_PER_SPOT_REQUEST, NETWORK, PROJECT_ID, QUOTA, RUNTIME_VERSION, ZONE_ABBREV, account_flags

# gcloud --accelerator-type prefixes. v5e is the odd one out: its pod sizes
# are named "v5litepod-N", not "v5e-N".
ACCEL_PREFIX = {
    "v4": "v4",
    "v5e": "v5litepod",
    "v5p": "v5p",
    "v6e": "v6e",
}


def accelerator_type(generation: str, chips: int) -> str:
    prefix = ACCEL_PREFIX.get(generation)
    if prefix is None:
        raise ValueError(f"unknown TPU generation: {generation!r}")
    return f"{prefix}-{chips}"


def split_entry(entry: dict) -> list[dict]:
    """Split one quota entry into one or more per-request chip counts."""
    chips, generation, tier = entry["chips"], entry["generation"], entry["tier"]

    per_request = MAX_CHIPS_PER_SPOT_REQUEST.get(generation, chips) if tier == "spot" else chips

    if chips % per_request != 0:
        raise ValueError(
            f"{entry['zone']} {generation} {tier}: {chips} chips is not a "
            f"multiple of the {per_request}-chip split size. Fix "
            f"MAX_CHIPS_PER_SPOT_REQUEST in trc_quota.py or split this "
            f"entry by hand."
        )

    num_requests = chips // per_request
    return [{**entry, "chips": per_request} for _ in range(num_requests)]


def slice_name(entry: dict, index: int) -> str:
    zone_abbrev = ZONE_ABBREV.get(entry["zone"])
    if zone_abbrev is None:
        raise ValueError(f"no ZONE_ABBREV entry for zone {entry['zone']!r} -- add one in trc_quota.py")
    return f"trc-{entry['generation']}-{entry['chips']}-{zone_abbrev}-{entry['tier']}-{index}"


def region_of(zone: str) -> str:
    """"us-central1-a" -> "us-central1" (drop the trailing zone letter)."""
    return zone.rsplit("-", 1)[0]


def render_command(cmd: list) -> str:
    """Shell-quote an argv list for display/copy-paste.

    Plain `" ".join(cmd)` breaks for arguments like
    `--format=value(name,state)` -- pasted into a real shell, the
    unquoted `(` is parsed as subshell syntax. shlex.join quotes whatever
    needs it so the printed command is safe to paste and run as-is.
    """
    return shlex.join(cmd)


def build_command(entry: dict, index: int) -> list[str]:
    generation, zone, chips, tier = (
        entry["generation"],
        entry["zone"],
        entry["chips"],
        entry["tier"],
    )
    name = slice_name(entry, index)
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--node-id={name}",
        f"--accelerator-type={accelerator_type(generation, chips)}",
        f"--runtime-version={RUNTIME_VERSION[generation]}",
    ]
    if tier == "spot":
        cmd.append("--spot")
    if tier != "on-demand":
        cmd.append("--internal-ips")
    cmd.extend(account_flags())
    return cmd


def build_list_command(zone: str) -> list[str]:
    """Check what queued resources actually exist in `zone` right now."""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--format=value(name,state)",
    ]
    cmd.extend(account_flags())
    return cmd


def build_describe_command(name: str, zone: str) -> list[str]:
    """Full detail for one queued resource -- list only gives name+state.

    Useful for digging into *why* a specific slice is stuck (e.g. a spot
    request that's been WAITING_FOR_RESOURCES for hours) instead of just
    knowing that it is.
    """
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
    ]
    cmd.extend(account_flags())
    return cmd


def build_delete_command(name: str, zone: str) -> list[str]:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "delete",
        name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        "--quiet",
        "--force",
    ]
    cmd.extend(account_flags())
    return cmd


def build_enable_private_google_access_command(region: str) -> list[str]:
    """One-time per-region prerequisite for --internal-ips (see build_command).

    Without this, `queued-resources create` fails with INVALID_ARGUMENT
    because a TPU with no external IP can't reach Google's own APIs to
    provision itself unless the subnet explicitly allows it.
    """
    cmd = [
        "gcloud",
        "compute",
        "networks",
        "subnets",
        "update",
        NETWORK,
        f"--project={PROJECT_ID}",
        f"--region={region}",
        "--enable-private-ip-google-access",
    ]
    cmd.extend(account_flags())
    return cmd


def build_nat_router_command(region: str) -> list[str]:
    """First half of one-time per-region Cloud NAT setup (see build_nat_gateway_command).

    Separate from Private Google Access: this is for the TPU VM reaching the
    general internet (pip install, downloading models/datasets) once it's
    running, not for the create call itself to succeed.
    """
    cmd = [
        "gcloud",
        "compute",
        "routers",
        "create",
        f"nat-router-{region}",
        f"--project={PROJECT_ID}",
        f"--network={NETWORK}",
        f"--region={region}",
    ]
    cmd.extend(account_flags())
    return cmd


def build_nat_gateway_command(region: str) -> list[str]:
    """Second half of one-time per-region Cloud NAT setup -- run after build_nat_router_command."""
    cmd = [
        "gcloud",
        "compute",
        "routers",
        "nats",
        "create",
        f"nat-config-{region}",
        f"--project={PROJECT_ID}",
        f"--router=nat-router-{region}",
        f"--region={region}",
        "--auto-allocate-nat-external-ips",
        "--nat-all-subnet-ip-ranges",
    ]
    cmd.extend(account_flags())
    return cmd


def _command_block(items: list) -> list:
    """Render [(label, cmd), ...] as one ```bash``` block, one `# label` comment per command.

    All commands for a section share a single code block (so it reads like a
    normal shell script and stays compact in a plain editor) instead of one
    block per command; the `# label` line above each command is what makes
    a specific one findable when scanning or diffing. GitHub still gives you
    a copy button, it just copies the whole section rather than one line.
    """
    lines = ["```bash"]
    for label, cmd in items:
        lines.append(f"# {label}")
        lines.append(render_command(cmd))
    lines.append("```")
    lines.append("")
    return lines


def write_command_file(
    path: str,
    pga_items: list,
    nat_items: list,
    check_items: list,
    create_items: list,
    describe_items: list,
    delete_spot_items: list,
    delete_on_demand_items: list,
) -> None:
    """Write a Markdown command reference: network setup, check, create, describe, then delete.

    All *_items args are [(label, cmd), ...], where label is a region
    (network setup, check) or slice name (create/describe/delete) and cmd is
    the gcloud argv list. This is a reference to copy commands out of, not a
    script meant to be run top-to-bottom -- the delete section targets the
    same slice names the create section makes.

    Delete is split into spot/on-demand subsections rather than one combined
    list: spot slices get deleted+recreated routinely (that's the whole
    point of reconcile.py -- spot capacity gets preempted), but on-demand
    slices don't get preempted and aren't meant to be torn down as part of
    that routine. Keeping them in a visibly separate section makes it harder
    to absent-mindedly copy an on-demand delete command along with a batch
    of spot ones.
    """
    lines = [
        "# TPU provisioning commands",
        "",
        "Auto-generated by `plan_tpu_requests.py` -- do not hand-edit, rerun the "
        "script to regenerate. Sections are independent (e.g. don't paste the "
        "whole Create section and then the whole Delete section back-to-back -- "
        "that just tears down what it made).",
        "",
        "## Network Setup",
        "",
        "One-time per-region setup, not per-TPU -- run once per region, not on "
        "every create.",
        "",
        "### Private Google Access",
        "",
        "Required for `--internal-ips` create commands to succeed (see "
        "[README.md 注意事項](README.md#注意事項)): a TPU with no external IP "
        "still needs to reach Google's own APIs to provision itself.",
        "",
        *_command_block(pga_items),
        "### Cloud NAT",
        "",
        "For the TPU VM to reach the general internet (pip install, downloading "
        "models/datasets) once it's running -- separate concern from Private "
        "Google Access above. Each region needs the router command run before "
        "the nat command.",
        "",
        *_command_block(nat_items),
        "## Check",
        "",
        "What queued resources currently exist, per zone.",
        "",
        *_command_block(check_items),
        "## Create",
        "",
        "Request the slices in `trc_quota.py`'s `QUOTA`.",
        "",
        *_command_block(create_items),
        "## Describe",
        "",
        "Full detail for one slice (list only gives name + state) -- useful for "
        "digging into why a specific slice is stuck.",
        "",
        *_command_block(describe_items),
        "## Delete",
        "",
        "Remove a slice created above.",
        "",
        "### Spot",
        "",
        "Spot slices get preempted routinely -- deleting and letting the next "
        "create/reconcile pass rebuild them is normal, expected operation.",
        "",
        *_command_block(delete_spot_items),
        "### On-demand",
        "",
        "On-demand slices don't get preempted, so there's normally no reason "
        "to delete one as part of routine cleanup -- these commands are here "
        "for when you deliberately want to tear one down, not for regular use.",
        "",
        *_command_block(delete_on_demand_items),
    ]
    Path(path).write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="execute the create commands instead of just printing them")
    parser.add_argument("--yes", action="store_true", help="with --run, skip the confirmation prompt")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "tpu_commands.md"),
        help="where to write the check/create/delete command reference (default: %(default)s)",
    )
    args = parser.parse_args()

    if not PROJECT_ID:
        sys.exit("Set the TPU_PROJECT_ID environment variable before running this script, e.g.:\n" "  export TPU_PROJECT_ID=my-gcp-project")

    plan = []  # (entry_with_split_chips, per_request_commands, names) per quota row
    for entry in QUOTA:
        requests = split_entry(entry)
        commands = [build_command(req, i) for i, req in enumerate(requests)]
        names = [slice_name(req, i) for i, req in enumerate(requests)]
        plan.append((entry, requests, commands, names))

    print("# Plan summary: zone / generation / tier -> num_requests x chips_per_request (quota)")
    all_commands = []
    all_slices = []  # (name, zone, tier) for every slice, used to build delete commands
    for entry, requests, commands, names in plan:
        n = len(requests)
        per = requests[0]["chips"]
        print(f"#   {entry['zone']:<16} {entry['generation']:<4} {entry['tier']:<10} " f"{n} x {per:<3} = {n * per:<4} (quota: {entry['chips']})")
        all_commands.extend(commands)
        all_slices.extend((name, req["zone"], req["tier"]) for name, req in zip(names, requests))
    print()

    for cmd in all_commands:
        print(render_command(cmd))

    zones = sorted({entry["zone"] for entry in QUOTA})
    regions = sorted({region_of(zone) for zone in zones})
    pga_items = [(region, build_enable_private_google_access_command(region)) for region in regions]
    nat_items = []
    for region in regions:
        nat_items.append((region, build_nat_router_command(region)))
        nat_items.append((region, build_nat_gateway_command(region)))
    check_items = [(zone, build_list_command(zone)) for zone in zones]
    create_items = list(zip((name for name, _, _ in all_slices), all_commands))
    describe_items = [(name, build_describe_command(name, zone)) for name, zone, _ in all_slices]
    delete_spot_items = [(name, build_delete_command(name, zone)) for name, zone, tier in all_slices if tier == "spot"]
    delete_on_demand_items = [(name, build_delete_command(name, zone)) for name, zone, tier in all_slices if tier == "on-demand"]
    write_command_file(
        args.output, pga_items, nat_items, check_items, create_items, describe_items, delete_spot_items, delete_on_demand_items
    )
    print(f"\nFull network-setup/check/create/describe/delete command reference written to {args.output}")

    if not args.run:
        return

    if not args.yes:
        reply = input(f"\nRun the {len(all_commands)} create commands above? [y/N] ")
        if reply.strip().lower() != "y":
            print("Aborted.")
            return

    for cmd in all_commands:
        print(f"\n$ {render_command(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
