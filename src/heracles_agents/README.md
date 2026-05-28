# heracles_agents

`heracles_agents` contains the agent framework, provider integrations, tools,
and experiment pipelines used to evaluate language-model interfaces for 3D
scene graph question answering.

Package resources such as grammar and label-space data live in `resources/`.
Provider-specific code is grouped under `provider_integrations/`, reusable
agent tools under `tools/`, shared tool-call infrastructure under
`tool_calling/`, command-line utilities under `cli/`, and runnable evaluation
flows under `pipelines/`.
