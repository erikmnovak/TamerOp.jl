#!/usr/bin/env bash
set -euo pipefail

# Ubuntu restricts user namespaces for downloaded Chromium executables. Allow
# only the installed Playwright binaries to create their own browser sandboxes.
# See https://chromium.googlesource.com/chromium/src/+/main/docs/security/apparmor-userns-restrictions.md
if [[ ${GITHUB_ACTIONS:-} != true || ${RUNNER_OS:-} != Linux ]]; then
    echo 'This setup is intended only for the Linux GitHub Actions runner.' >&2
    exit 1
fi

restriction=/proc/sys/kernel/apparmor_restrict_unprivileged_userns
if [[ ! -f $restriction ]] || [[ $(cat "$restriction") != 1 ]]; then
    echo 'AppArmor does not restrict unprivileged user namespaces on this runner.'
    exit 0
fi

browser_root=${PLAYWRIGHT_BROWSERS_PATH:-${XDG_CACHE_HOME:-$HOME/.cache}/ms-playwright}
shopt -s nullglob
chromium=("$browser_root"/chromium-*/chrome-linux64/chrome)
headless=("$browser_root"/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell)
if (( ${#chromium[@]} == 0 || ${#headless[@]} == 0 )); then
    echo 'Install both Playwright Chromium builds before configuring AppArmor.' >&2
    exit 1
fi

profile=$(mktemp)
trap 'rm -f "$profile"' EXIT
printf 'abi <abi/4.0>,\ninclude <tunables/global>\n' > "$profile"
index=0
for executable in "${chromium[@]}" "${headless[@]}"; do
    executable=$(realpath "$executable")
    # AppArmor paths are quoted, and glob/profile metacharacters are rejected.
    if [[ ! -x $executable || ! $executable =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
        echo "Unsupported Chromium executable path: $executable" >&2
        exit 1
    fi
    printf 'profile tamerop-playwright-%s "%s" flags=(unconfined) {\n  userns,\n}\n' \
        "$index" "$executable" >> "$profile"
    index=$((index + 1))
done
sudo install -m 0644 "$profile" /etc/apparmor.d/tamerop-playwright
sudo apparmor_parser -r /etc/apparmor.d/tamerop-playwright
echo 'AppArmor permits user namespaces for the installed Playwright Chromium binaries.'
