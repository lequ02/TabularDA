# Checkpoint secret scan and cleanup

Date: 2026-09-18. Scanner: Gitleaks 8.30.1, verified against its official release checksum. Reports redact detected values.

## Findings

- The published checkpoint imported Google API key strings in 28 saved Google Drive HTML pages. Those pages were fetched from public Google Drive URLs without an authorization header. The parsed folder JSON listings retain the research evidence; the raw browser application pages are omitted from the cleaned history.
- Eight GitHub metadata JSON files included signed raw-content download URLs with token query parameters. Those parameters are redacted in every cleaned checkpoint commit.
- The remaining generic-key matches were test accuracy/F1/loss metric identifiers. The scanner configuration excludes only those exact metric name patterns; credential rules remain enabled.
- No findings were detected in the new GLRM or pyglrm commits.

## Prevention

The Drive reader saves parsed folder listings instead of raw HTML. Ignore rules exclude new raw Drive HTML snapshots. The .gitleaks.toml file extends the default rules and documents the narrow metric-name exclusion.

## Verification scope

Scan the replacement checkpoint history with:

    gitleaks git --redact=100 --max-target-megabytes 32 --log-opts '248cb9b..HEAD' .

This scans every changed commit in the replacement checkpoint, not all pre-existing repository history, ignored local datasets, LFS binary contents, or external clones/caches. No credential validity or ownership was tested. No captured key was used to contact an API.

## Publication status

This replacement reconstructs the affected checkpoint commits without the captured credentials. The user authorized replacement of the published checkpoint branch and tag. GLRM and pyglrm backup references contain no detected credentials and need no replacement. Removing the affected commits from advertised branches and tags does not erase GitHub SHA-based cached views or other clones; GitHub Support must handle server-side garbage collection and cache removal.

If an exposed credential belongs to you, revoke or rotate it. See https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository for cache/history cleanup guidance.

## Verified results

The replacement checkpoint commit range passed Gitleaks with zero findings, using the default rules plus the exact metric-name exclusion. The GLRM and pyglrm commit ranges also passed with zero findings. A mocked Drive download regression check confirmed that parsed listings are identical and no raw HTML is written.

A separate streaming pattern scan covered 26 changed binary files and both decompressed members of the model ZIP (751,691,483 bytes total). It found no common Google/GitHub/OpenAI/Slack keys, AWS access IDs, private key headers, or signed URL tokens. Binary-pattern scanning does not establish that no other credential formats exist.
