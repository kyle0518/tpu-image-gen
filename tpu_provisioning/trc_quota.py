"""TRC quota grant + known-good Queued Resource split sizes.

Update QUOTA every time the TPU Research Cloud grant is renewed or changed
(copy the numbers straight from the approval email). `generation` is one of
"v4", "v5e", "v5p", "v6e" (see ACCEL_PREFIX in plan_tpu_requests.py for how
each maps to a gcloud --accelerator-type value). `tier` is "spot" or
"on-demand".

PROJECT_ID and GCLOUD_ACCOUNT identify *who* the gcloud commands act as, so
they come from the environment instead of being hardcoded here -- that way
this file can be committed/shared without leaking which GCP project/account
it targets. Zones, generations, and chip counts (below) aren't credentials,
so those stay hardcoded.
"""

import os

PROJECT_ID = os.environ.get("TPU_PROJECT_ID")  # required; export before running

# Optional. If unset, gcloud commands use whatever account `gcloud auth
# login` last activated. Set this to pin every generated command to a
# specific account (e.g. a shared service account) instead of relying on
# whatever happens to be active on the machine that runs cron.
GCLOUD_ACCOUNT = os.environ.get("TPU_GCLOUD_ACCOUNT")


def account_flags() -> list[str]:
    """--account=... flag to append to a gcloud command, or [] if unset."""
    return [f"--account={GCLOUD_ACCOUNT}"] if GCLOUD_ACCOUNT else []


# VPC network the TPU create commands implicitly use (they don't pass
# --network, so GCP defaults to this one) -- also the network the Private
# Google Access / Cloud NAT setup commands need to target so they actually
# affect the subnet the TPUs land in.
NETWORK = "default"


QUOTA = [
    {"zone": "europe-west4-b", "generation": "v5e", "tier": "spot", "chips": 64},
    {"zone": "us-east1-d", "generation": "v6e", "tier": "spot", "chips": 64},
    {"zone": "us-central1-a", "generation": "v5e", "tier": "spot", "chips": 64},
    {"zone": "us-central2-b", "generation": "v4", "tier": "spot", "chips": 32},
    {"zone": "us-central2-b", "generation": "v4", "tier": "on-demand", "chips": 32},
    {"zone": "europe-west4-a", "generation": "v6e", "tier": "spot", "chips": 64},
]

# Short form of each zone used in generated slice names (see slice_name() in
# plan_tpu_requests.py), e.g. "us-central1-a" -> "uscent1a". Hardcoded per
# zone rather than derived algorithmically -- add an entry here whenever a
# new zone is added to QUOTA above.
ZONE_ABBREV = {
    "europe-west4-a": "euwest4a",
    "europe-west4-b": "euwest4b",
    "us-central1-a": "uscent1a",
    "us-central2-b": "uscent2b",
    "us-east1-d": "useast1d",
}

# Empirically observed: requesting a single Queued Resource at the full spot
# quota size (e.g. v5e-64, v6e-64 in one QR) has failed to get fulfilled for
# this project. Splitting the same total chip count into this many chips per
# request, issued as several separate QueuedResources in the same zone, has
# worked instead (v5e-32 x2, v6e-16 x4). Re-verify these numbers if TRC
# capacity behavior changes -- they are empirical, not a documented API limit.
#
# The v4 grant is small enough (32 chips) that it has only ever been
# requested as a single QR, so its entry below is an untested default, not a
# confirmed limit.
MAX_CHIPS_PER_SPOT_REQUEST = {
    "v5e": 32,
    "v6e": 16,
    "v4": 32,
}

# Sanity-check these against `gcloud compute tpus versions list --zone=<zone>`
# before relying on them -- runtime version identifiers change over time and
# these are best-effort defaults, not guaranteed current.
RUNTIME_VERSION = {
    "v4": "tpu-ubuntu2204-base",
    "v5e": "v2-alpha-tpuv5-lite",
    "v6e": "v2-alpha-tpuv6e",
}
