---
title: Always Verify Cron Job Execution After Creation
severity: medium
category: orchestration
tags: [cron, verification, testing]
created: 2026-01-31
---

## Rule
After creating cron jobs, always run a manual trigger or wait for first execution to verify they work correctly.

## Context
During system verification, discovered that orchestration cron jobs were created but showed "-" for last execution, meaning they hadn't run yet. This could lead to assuming features are working when they're just scheduled but not proven.

## Implementation
1. After creating cron job, note the next scheduled time
2. Either wait for execution or manually trigger if testing
3. Check logs/output to confirm successful execution
4. Document any issues with timing or permissions

## Example
```bash
# Create cron
clawdbot cron add --name "test" --schedule "*/5 * * * *" --task "echo test"

# Wait 5 minutes or manually execute
clawdbot cron trigger <cron-id>

# Verify execution
clawdbot cron list | grep test
```