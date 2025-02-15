# check for Dafny and Z3

DAFNY_VERSION="4.9.0"

if ! which dafny > /dev/null 2>&1; then
    echo "Dafny version ${DAFNY_VERSION} needs to be in your path."
    exit 1
fi

INSTALLED_DAFNY_VERSION=$( dafny --version | cut -d'+' -f1 )
if [[ "$INSTALLED_DAFNY_VERSION" != "$DAFNY_VERSION" ]]; then
    echo "Installed Dafny version is $INSTALLED_DAFNY_VERSION but we need $DAFNY_VERSION"
    exit 1
fi

Z3_VERSION="4.13.4"

if ! which z3 > /dev/null 2>&1; then
    echo "Z3 version ${Z3_VERSION} needs to be in your path."
    exit 1
fi

INSTALLED_Z3_VERSION=$(z3 --version | cut -d' ' -f3)
if [[ "$INSTALLED_Z3_VERSION" != "$Z3_VERSION" ]]; then
    echo "Installed Z3 version is '$INSTALLED_Z3_VERSION' but we need '$Z3_VERSION'"
    exit 1
fi

Z3_PATH=$(which z3)

FILES=(basic_arithmetic linear_algebra main neural_networks parsing robustness_certification l2_extra)
ISOLATE_ASSERTIONS_FILES=(operator_norms)

for i in ${FILES[@]}
do
    echo Verifying $i.dfy...
    echo "dafny verify $i.dfy --solver-path ${Z3_PATH}"
    dafny verify $i.dfy --solver-path "${Z3_PATH}"
done

for i in ${ISOLATE_ASSERTIONS_FILES[@]}
do
    echo Verifying $i.dfy...
    echo "dafny verify $i.dfy --isolate-assertions --solver-path ${Z3_PATH}"
    dafny verify $i.dfy --isolate-assertions --solver-path "${Z3_PATH}"
done
