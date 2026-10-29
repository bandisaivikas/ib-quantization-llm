# Security Policy

## Supported Versions

Only the latest version on the `main` branch is supported.

## Reporting a Vulnerability

Please **do not** open a public issue for security problems.

Instead, use GitHub's private reporting:

1. Go to the **Security** tab of this repository.
2. Click **Report a vulnerability**.
3. Describe the issue and how to reproduce it.

You can expect an initial response within 7 days.

## Scope

This is a research codebase. Relevant issues include:

- Code that could leak HuggingFace tokens or other credentials
- Unsafe loading of untrusted model files or pickles
- Dependencies with known vulnerabilities

## Good Practice for Users

- Never hard-code your HuggingFace token in notebooks. Use `huggingface-cli login` or an environment variable.
- Only load model weights from trusted sources.
