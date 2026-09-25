# Security policy

Do not commit credentials, GitHub tokens, portal cookies, challenge data, or
private organizer materials. Report a suspected vulnerability privately to the
repository owner rather than opening a public issue containing sensitive
information.

The pipeline accepts local challenge TSVs and does not require network access at
runtime. SQLite work directories should be on a local filesystem with reliable
locking; do not run concurrent writers on SMB, NFS, or cloud-synchronized
folders.
