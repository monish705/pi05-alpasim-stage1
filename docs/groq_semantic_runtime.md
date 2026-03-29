# Groq Semantic Runtime Setup

This runbook records the exact multimodal setup we are using for the semantic autonomy layer before wiring it into the CARLA A-to-B mission loop.

## Scope

- Provider: Groq
- API style: OpenAI-compatible `chat.completions`
- Model: `meta-llama/llama-4-scout-17b-16e-instruct`
- Current purpose: image-grounded tactical decision on a single frame
- Later purpose: run on sampled CARLA chase-camera or ego-camera frames during an A-to-B mission

## Official Compatibility Notes

Verified against Groq's current vision docs on March 27, 2026:

- `meta-llama/llama-4-scout-17b-16e-instruct` supports text + image input
- JSON mode is supported with `response_format={"type": "json_object"}`
- local files can be sent as base64 data URLs inside `image_url`
- the documented path is `chat.completions`

Reference:

- [Groq vision docs](https://console.groq.com/docs/vision)

## Local Repo Files

- client: `sim/groq_semantic_client.py`
- smoke test CLI: `sim/groq_semantic_smoke.py`
- default artifact output: `artifacts/groq_semantic/`

## Runtime Contract

The semantic layer currently emits one of these tactical actions:

- `follow_route`
- `yield`
- `change_lane_left`
- `change_lane_right`
- `creep_forward`
- `reroute`

The current JSON response schema is:

```json
{
  "scene_summary": "string",
  "risk_level": "low|medium|high",
  "recommended_action": "follow_route|yield|change_lane_left|change_lane_right|creep_forward|reroute",
  "reason": "string",
  "confidence": 0.0
}
```

## Local Smoke Test

Set the API key in the shell, then run:

```powershell
$env:GROQ_API_KEY="..."
py -3.13 sim\groq_semantic_smoke.py `
  --image test_frame.png `
  --mission "Drive from the marked start point to the marked destination while staying safe."
```

This writes a timestamped JSON artifact under:

```text
artifacts/groq_semantic/
```

## Later CARLA Integration Path

The intended next integration is:

1. capture a CARLA frame during a running mission
2. summarize route state and nearby-actor context as text
3. call `GroqSemanticClient.semantic_decision(...)`
4. map the returned semantic action into low-level behavior controls
5. append the decision to the mission trace JSON

This keeps the architecture clean:

- low-level driver handles controls
- semantic runtime handles tactical intent
- CARLA remains the execution and evaluation environment
