# livekitstaging

## Outbound agent worker (Cloud Run)

This repository contains a LiveKit worker process (not an HTTP function). Deploy it as a
containerized Cloud Run service, then dispatch calls through LiveKit using explicit dispatch.

### Runtime routing model

- One deployed worker image can support multiple business workflows.
- `agent_id` from dispatch metadata selects the runtime behavior.
- Firestore can override routing with `agent_kind`.

Supported runtime kinds in this codebase:

- `sales`
- `collections`
- `retention`

### Required service shape

- Cloud Run service type: **container service**
- Authentication: **require authentication**
- CPU allocation: **instance-based billing**
- Max concurrent requests per instance: **1**
- Timeout: **900s** (or higher if your calls run longer)
- Min instances: **>=1** to reduce cold starts

### Deploy steps

1. Ensure Artifact Registry repository exists.
2. Build and deploy with helper script:

```bash
chmod +x scripts/deploy_cloud_run_worker.sh
./scripts/deploy_cloud_run_worker.sh <gcp_project_id> <region> <artifact_repo> [service_name] [image_tag]
```

3. Configure secrets and env vars in Cloud Run revision.
4. In your trigger/orchestration service, dispatch using:
   - `agent_name` = worker registration name
   - metadata includes `client_id`, `event_id`, `agent_id`, and contact payload

### Environment variables

See `.env.cloudrun.example` for a baseline template.