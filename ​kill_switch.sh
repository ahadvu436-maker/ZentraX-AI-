#!/usr/bin/env bash
#
# kill_switch.sh - instantly cut a server off from the internet while keeping
# loopback, LAN and other private-network traffic working.
#
#   sudo ./kill_switch.sh on  [--revert-after SECS] [--force] [--dry-run]
#   sudo ./kill_switch.sh off [--dry-run]
#        ./kill_switch.sh status
#
# Design notes
#  * It does NOT shut the firewall down. Disabling a firewall opens the server up,
#    the opposite of isolation. Instead it adds a self-contained rule set that
#    drops every non-local packet (in, out and forwarded, including
#    already-established internet connections) and removes cleanly on "off".
#  * Existing firewall config (ufw, iptables, nftables, cloud agents) is not
#    modified, so "off" restores exactly what was there before.
#  * nftables is preferred (one atomic ruleset load); iptables/ip6tables is the
#    fallback. Override with KS_BACKEND=nft|iptables.
#  * Rules are runtime-only: a reboot also lifts the block.
#
# Environment
#  LOCAL_NETS   space-separated CIDRs that stay reachable (default: loopback,
#               RFC1918, link-local, ULA, broadcast/multicast needed by DHCP/NDP)
#  ALLOW_EXTRA  extra CIDRs to keep reachable, e.g. a bastion's public IP:
#               ALLOW_EXTRA="203.0.113.10/32"

set -Eeuo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
LOCK=/run/kill_switch.lock
NFT_TABLE=killswitch
CHAINS=(KS_IN KS_OUT KS_FWD)

LOCAL_NETS="${LOCAL_NETS:-127.0.0.0/8 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 169.254.0.0/16 0.0.0.0/32 255.255.255.255/32 ::1/128 ::/128 fc00::/7 fe80::/10 ff02::/16}"
ALLOW_EXTRA="${ALLOW_EXTRA:-}"

DRY=0; FORCE=0; REVERT=0
V4=(); V6=(); BACKEND=''; FN=''

die()  { echo "kill_switch: ERROR: $*" >&2; exit 1; }
warn() { echo "kill_switch: warning: $*" >&2; }
info() { echo "kill_switch: $*"; logger -t kill_switch -- "$*" 2>/dev/null || true; }
trap 'echo "kill_switch: failed at line $LINENO" >&2' ERR

# ---------------------------------------------------------------- helpers
split_nets() {
  local n
  for n in $LOCAL_NETS $ALLOW_EXTRA; do
    if [[ $n == *:* ]]; then V6+=("$n"); else V4+=("$n"); fi
  done
}

join_comma() { local IFS=,; echo "$*"; }

ip2int() { local IFS=.; local a b c d; read -r a b c d <<<"$1"; echo $(( (a<<24)|(b<<16)|(c<<8)|d )); }

in_cidr() {   # in_cidr 10.1.2.3 10.0.0.0/8
  local ip=$1 net=${2%/*} bits=${2#*/} mask
  mask=$(( bits == 0 ? 0 : (0xFFFFFFFF << (32 - bits)) & 0xFFFFFFFF ))
  (( ($(ip2int "$ip") & mask) == ($(ip2int "$net") & mask) ))
}

detect_backend() {
  if [[ -n ${KS_BACKEND:-} ]]; then BACKEND=$KS_BACKEND
  elif command -v nft >/dev/null 2>&1; then BACKEND=nft
  elif command -v iptables >/dev/null 2>&1; then BACKEND=iptables
  else die "neither nft nor iptables found"; fi
  case $BACKEND in
    nft)      FN=nft ;;
    iptables) FN=ipt ;;
    *) die "KS_BACKEND must be nft or iptables" ;;
  esac
  [[ $DRY -eq 1 ]] || command -v "$BACKEND" >/dev/null 2>&1 || die "$BACKEND not installed"
}

need_root() { [[ $DRY -eq 1 || $EUID -eq 0 ]] || die "must run as root (try sudo)"; }

lock() {
  [[ $DRY -eq 1 ]] && return 0
  exec 9>"$LOCK"
  flock -n 9 || die "another kill_switch.sh instance is running"
}

# Refuse to cut off the admin's own SSH session unless they allowed it.
preflight_ssh() {
  [[ $FORCE -eq 1 || -z ${SSH_CONNECTION:-} ]] && return 0
  local client=${SSH_CONNECTION%% *} n
  if [[ ! $client =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    warn "SSH client $client is not IPv4; make sure it is covered by LOCAL_NETS/ALLOW_EXTRA"
    return 0
  fi
  for n in "${V4[@]}"; do in_cidr "$client" "$n" && return 0; done
  die "your SSH session ($client) is outside the allowed networks and would be cut. Add it with ALLOW_EXTRA=\"$client/32\", or use --force"
}

# ---------------------------------------------------------------- nftables
nft_ruleset() {
  local v4 v6 sets="" in4="" in6="" out4="" out6="" fwd4="" fwd6=""
  v4=$(join_comma "${V4[@]}"); v6=$(join_comma "${V6[@]}")
  if [[ -n $v4 ]]; then
    sets+="  set local4 { type ipv4_addr; flags interval; auto-merge; elements = { $v4 } }"$'\n'
    in4="ip saddr @local4 accept"; out4="ip daddr @local4 accept"
    fwd4="ip saddr @local4 ip daddr @local4 accept"
  fi
  if [[ -n $v6 ]]; then
    sets+="  set local6 { type ipv6_addr; flags interval; auto-merge; elements = { $v6 } }"$'\n'
    in6="ip6 saddr @local6 accept"; out6="ip6 daddr @local6 accept"
    fwd6="ip6 saddr @local6 ip6 daddr @local6 accept"
  fi
  cat <<EOF
add table inet $NFT_TABLE
delete table inet $NFT_TABLE
table inet $NFT_TABLE {
$sets  chain in {
    type filter hook input priority -300; policy accept;
    iifname "lo" accept
    $in4
    $in6
    counter drop
  }
  chain out {
    type filter hook output priority -300; policy accept;
    oifname "lo" accept
    $out4
    $out6
    counter drop
  }
  chain fwd {
    type filter hook forward priority -300; policy accept;
    $fwd4
    $fwd6
    counter drop
  }
}
EOF
}

nft_on() {
  if [[ $DRY -eq 1 ]]; then nft_ruleset; else nft_ruleset | nft -f -; fi
}
nft_off() {
  if [[ $DRY -eq 1 ]]; then echo "nft delete table inet $NFT_TABLE"; return 0; fi
  nft delete table inet "$NFT_TABLE" 2>/dev/null || true
}
nft_active() { nft list table inet "$NFT_TABLE" >/dev/null 2>&1; }
nft_status() { nft list table inet "$NFT_TABLE" 2>/dev/null | grep -E 'counter|chain' || true; }

# ---------------------------------------------------------------- iptables
ipt_restore_text() {   # $1 = family (4|6)
  local -a nets; local n
  if [[ $1 == 4 ]]; then nets=("${V4[@]}"); else nets=("${V6[@]}"); fi
  echo "*filter"
  printf ':%s - [0:0]\n' "${CHAINS[@]}"
  printf -- '-F %s\n' "${CHAINS[@]}"
  echo "-A KS_IN -i lo -j ACCEPT"
  echo "-A KS_OUT -o lo -j ACCEPT"
  for n in "${nets[@]}"; do
    echo "-A KS_IN -s $n -j ACCEPT"
    echo "-A KS_OUT -d $n -j ACCEPT"
    echo "-A KS_FWD -s $n -d $n -j ACCEPT"
  done
  echo "-A KS_IN -j DROP"
  echo "-A KS_OUT -j DROP"
  echo "-A KS_FWD -j DROP"
  echo "COMMIT"
}

ipt_bins() { echo "iptables"; command -v ip6tables >/dev/null 2>&1 && echo "ip6tables"; return 0; }

ipt_on() {
  local bin fam
  for bin in $(ipt_bins); do
    fam=4; [[ $bin == ip6tables ]] && fam=6
    if [[ $DRY -eq 1 ]]; then echo "# $bin-restore --noflush"; ipt_restore_text $fam; continue; fi
    ipt_restore_text $fam | "$bin-restore" --noflush -w 2>/dev/null || ipt_restore_text $fam | "$bin-restore" --noflush
    $bin -w -C INPUT   -j KS_IN  2>/dev/null || $bin -w -I INPUT   1 -j KS_IN
    $bin -w -C OUTPUT  -j KS_OUT 2>/dev/null || $bin -w -I OUTPUT  1 -j KS_OUT
    $bin -w -C FORWARD -j KS_FWD 2>/dev/null || $bin -w -I FORWARD 1 -j KS_FWD
  done
}

ipt_off() {
  local bin c
  for bin in $(ipt_bins); do
    if [[ $DRY -eq 1 ]]; then echo "# $bin: remove KS_* jumps and chains"; continue; fi
    while $bin -w -D INPUT   -j KS_IN  2>/dev/null; do :; done
    while $bin -w -D OUTPUT  -j KS_OUT 2>/dev/null; do :; done
    while $bin -w -D FORWARD -j KS_FWD 2>/dev/null; do :; done
    for c in "${CHAINS[@]}"; do $bin -w -F "$c" 2>/dev/null || true; $bin -w -X "$c" 2>/dev/null || true; done
  done
}
ipt_active() { iptables -w -C INPUT -j KS_IN >/dev/null 2>&1; }
ipt_status() { iptables -w -L KS_IN -n -v 2>/dev/null || true; }

# ---------------------------------------------------------------- commands
cmd_on() {
  split_nets
  preflight_ssh
  "${FN}_on"
  info "ISOLATED via $BACKEND: internet blocked; allowed: ${V4[*]:-} ${V6[*]:-}"
  if [[ $REVERT -gt 0 && $DRY -eq 0 ]]; then
    setsid -f bash -c 'sleep "$1"; exec "$2" off' _ "$REVERT" "$SELF" >/dev/null 2>&1
    info "auto-revert scheduled in ${REVERT}s"
  fi
}

cmd_off() {
  "${FN}_off"
  info "RESTORED: kill switch removed, previous firewall state untouched"
}

cmd_status() {
  if "${FN}_active"; then echo "kill switch: ACTIVE ($BACKEND)"; "${FN}_status"
  else echo "kill switch: inactive ($BACKEND)"; fi
}

usage() { sed -n '3,12p' "$SELF" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---------------------------------------------------------------- main
ACTION=${1:-}; [[ -n $ACTION ]] || usage 1
shift || true
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --force)   FORCE=1 ;;
    --revert-after) shift; REVERT=${1:-}; [[ $REVERT =~ ^[0-9]+$ ]] || die "--revert-after needs seconds" ;;
    -h|--help) usage 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

detect_backend
case $ACTION in
  on)     need_root; lock; cmd_on ;;
  off)    need_root; lock; cmd_off ;;
  status) cmd_status ;;
  -h|--help|help) usage 0 ;;
  *) usage 1 ;;
esac
