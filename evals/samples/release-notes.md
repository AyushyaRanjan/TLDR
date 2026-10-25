# Release notes for widget 2.4

Widget 2.4 adds a health check on port 8080 and refuses to start when that
port is taken. The previous version, 2.3, kept running and logged a warning
instead. Operators who pinned 2.3 should read the upgrade section before
they move.

Upgrade by replacing the binary and restarting the service. The notes do not
mention Docker. They do not mention a database migration. Configuration files
from 2.3 still load.

Known issue: the health check waits 15 seconds and then exits. There is no
retry. Backups of the data directory are not described.
