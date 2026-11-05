# Model registry lookup, sourced by the counterexample and re-verification runners
# (artifact DESIGN §7.3: one model table for every script). The registry is
# python-certifier/models/registry.tsv, a tab-separated table with header
#   key dataset kind net layers weights_csv isize eps gram_rq1 gram_all ref_rq1 ref_all
#   inputs_all inputs_dir bias bias_pos biases_txt
# where weights_csv is relative to this repository's root, ref_* to
# python-certifier/models/precomputed/, inputs_* to python-certifier/tests/, and "-" means
# none. The sourcing script must have set PC (python-certifier's root); REGISTRY overrides
# the path. Lines starting with # and blank lines are ignored; the first other line is the header.
#   reg KEY COLUMN   prints the cell ("-" prints as empty); fails if KEY or COLUMN is absent.
# Under set -e, assign with  v=$(reg k c)  (a plain assignment propagates the failure;
# "local v=$(...)" would mask it).
REGISTRY="${REGISTRY:-$PC/models/registry.tsv}"
reg () {
  local key="$1" col="$2" v
  [ -r "$REGISTRY" ] || { echo "registry: $REGISTRY not found (set REGISTRY)" >&2; return 1; }
  v=$(awk -F'\t' -v k="$key" -v c="$col" '
    /^#/ || /^[ \t]*$/ { next }                       # comment and blank lines
    !hdr  { hdr=1; for (i=1; i<=NF; i++) if ($i==c) ci=i; if (!ci) { print "\001NOCOL"; exit }; next }
    $1==k { print $ci; found=1; exit }
    END   { if (ci && !found) print "\001NOKEY" }' "$REGISTRY")
  case "$v" in
    $'\001NOCOL') echo "registry: no column '$col' in $REGISTRY" >&2; return 1 ;;
    $'\001NOKEY') echo "registry: no key '$key' in $REGISTRY" >&2; return 1 ;;
    -) v="" ;;
  esac
  printf '%s\n' "$v"
}
