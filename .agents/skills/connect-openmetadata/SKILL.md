---
name: connect-openmetadata
description: "Use when the user needs to authenticate against OpenMetadata / Metadata Hub (metadatahub.pg.com.cn), obtain or rotate a Personal Access Token (PAT / JWT), call OM REST API, or troubleshoot 401/403 errors on MetadataHub. Covers the Jira service-desk ticket flow, Bearer auth header, QA vs Prod base URLs, token expiry, and revocation."
---

# OpenMetadata Authentication & API Skill

## When to Use

Trigger this skill when the user or another agent:

- mentions **OpenMetadata**, **MetadataHub**, **metadatahub.pg.com.cn**, **PAT**, **Personal Access Token**, or **JWT**
- asks how to **申请 / 获取 / 轮换 token** for OM
- runs **schema / catalog 同步** or any task that hits the OM REST API
- encounters **401 Unauthorized** or **403 Forbidden** on a MetadataHub endpoint
- sets up a new dev box and needs to configure OM credentials

## Prerequisites

- OM user account (same identity that holds the PAT's permissions)
- Access to Jira service desk (`jira.pg.com.cn`)
- Password manager or secure secret store (never write PAT into repo/chat/email)

## Steps

### 1. Check for existing PAT

Look in this order:
1. Env var `OM_PAT` or `OPENMETADATA_TOKEN`
2. `~/.secrets/openmetadata_pat` (or equivalent password-manager entry)
3. Ask the user if nothing is found

### 2. Request a new PAT (admin-created, users cannot self-create)

Submit a Jira service-desk ticket:

- Portal: <https://jira.pg.com.cn/servicedesk/customer/portal/6/create/2963>
- Request type: **OpenMetadata User Token Create**
- Fields: OM username, purpose, requested validity (7 / 30 / 60 / 90 days — max 90)

Token is delivered via email or secure channel. **It is shown in full exactly once** — copy to password manager immediately; a lost token can only be revoked and re-issued, not recovered.

### 3. Pick the correct base URL

| Environment | Base URL |
|-------------|----------|
| QA          | `https://metadatahub-qa.pg.com.cn` |
| Prod        | `https://metadatahub.pg.com.cn` |

Do **not** append a trailing `/` before `/api/...`.

### 4. Use the PAT in requests

All authenticated OM endpoints take:

```http
Authorization: Bearer <PAT>
```

Smoke test:

```bash
curl -sS -X GET "https://metadatahub-qa.pg.com.cn/api/v1/users/loggedInUser" \
  -H "Authorization: Bearer $OM_PAT" \
  -H "Accept: application/json"
```

A 200 with the logged-in user JSON confirms auth works.

### 5. Using the PAT against OM REST API

Once `OM_PAT` is set, pass it as a `Bearer` token on every OM REST API call
(same pattern as the step 4 smoke test) to read catalog / schema metadata:

```bash
export OM_PAT=<token>         # or $env:OM_PAT=... on PowerShell
curl -sS -X GET "https://metadatahub-qa.pg.com.cn/api/v1/tables?limit=10" \
  -H "Authorization: Bearer $OM_PAT" \
  -H "Accept: application/json"
```

If a call fails with 401/403, re-run the step 4 smoke test to isolate auth vs. logic issues.

## Common Errors

| Symptom | Likely cause |
|---------|--------------|
| `401 Invalid personal access token` | token expired / revoked / copy truncated |
| `401` but token fresh | stray whitespace / newline after `Bearer` |
| `403 Forbidden` on specific endpoint | PAT identity lacks role/policy for that resource |
| `Connection refused` | wrong env (hitting prod URL from QA VPN or vice-versa) |

## Security Rules

1. Treat PAT as a password: never commit, screenshot, paste in chat, or email in plaintext.
2. Prefer a dedicated service account over a personal high-privilege account for scripts.
3. Use separate tokens for dev/QA/prod; revoke on role change or project end.
4. On suspected leak: request revoke + re-issue through the same Jira portal.

## Rotation

- Default validity: 7/30/60/90 days (admin choice, 90 max)
- Request new token via Jira **before** expiry; deactivate the old one once the new one is verified working.

## References

- OM API docs (login required): `https://metadatahub-qa.pg.com.cn/docs#overview` · `https://metadatahub.pg.com.cn/docs#overview`
- OpenMetadata official docs (v1.12.x): <https://docs.open-metadata.org/v1.12.x>
- Jira portal: <https://jira.pg.com.cn/servicedesk/customer/portal/6/create/2963>
