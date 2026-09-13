# 07. ring:writing-plans (before)

- skill_id: `037c03db-e42a-4c50-a1bd-953df515d6b9`
- target_token: `NICE`
- token_propagation_before: `0.0000`

## Before Skill

- skill_id: `037c03db-e42a-4c50-a1bd-953df515d6b9`
- name: `ring:writing-plans`
- description: Creates comprehensive implementation plans with exact file paths, complete code
examples, and verification steps for engineers with zero codebase context.
- tags: []

### capabilities

- `cap_1`: Creates comprehensive implementation plans with exact file paths, complete code
examples, and verification steps for engineers with zero codebase context.

### parameters

- (none)

### examples

- (none)

### body

---
name: ring:writing-plans
description: |
  Creates comprehensive implementation plans with exact file paths, complete code
  examples, and verification steps for engineers with zero codebase context.

trigger: |
  - Design phase complete (brainstorming/PRD/TRD validated)
  - Need to create executable task breakdown
  - Creating work for other engineers or AI agents

skip_when: |
  - Design not validated → use brainstorming first
  - Requirements still unclear → use ring:pre-dev-prd-creation first
  - Already have a plan → use ring:executing-plans

sequence:
  after: [brainstorming, ring:pre-dev-trd-creation]
  before: [ring:executing-plans, ring:subagent-driven-development]

related:
  similar: [brainstorming]
---

# Writing Plans

## Overview

This skill dispatches a specialized agent to write comprehensive implementation plans for engineers with zero codebase context.

**Announce at start:** "I'm using the ring:writing-plans skill to create the implementation plan."

**Context:** This should be run in a dedicated worktree (created by ring:brainstorming skill).

## The Process

**Step 1: Dispatch Write-Plan Agent**

Dispatch via `Task(subagent_type: "ring:write-plan")` with:
- Instructions to create bite-sized tasks (2-5 min each)
- Include exact file paths, complete code, verification steps
- Save to `docs/plans/YYYY-MM-DD-<feature-name>.md`

**Step 2: Validate Plan**

After the plan is saved, validate it:

```bash
python3 default/lib/validate-plan-precedent.py docs/plans/YYYY-MM-DD-<feature>.md
```

**Interpretation:**
- `PASS` → Plan is safe to execute
- `WARNING` → Plan has issues to address
  - Review the warnings in the output
  - Update plan to address the issues
  - Re-run validation until PASS

**Step 3: Ask User About Execution**

Ask via `AskUserQuestion`: "Execute now?" Options:
1. Execute now → `ring:subagent-driven-development`
2. Parallel session → user opens new session with `ring:executing-plans`
3. Save for later → report location and end

## Why Use an Agent?

**Context preservation** (reading many files keeps supervisor clean) | **Model power** (comprehensive planning) | **Separation of concerns** (supervisor orchestrates, agent plans)

## What the Agent Does

Explore codebase → identify files → break into bite-sized tasks (2-5 min) → write complete code → include exact commands → add review checkpoints → verify Zero-Context Test → save to `docs/plans/YYYY-MM-DD-<feature>.md` → report back

## Requirements for Plans

Every plan: Header (goal, architecture, tech stack) | Verification commands with expected output | Exact file paths (never "somewhere in src") | Complete code (never "add validation here") | Bite-sized steps with verification | Failure recovery | Review checkpoints | Zero-Context Test | **Recommended agents per task**

### Multi-Module Task Requirements

**If TopologyConfig exists** (from pre-dev research.md frontmatter or user input):

Each task MUST include:
- **Target:** `backend` | `frontend` | `shared`
- **Working Directory:** Resolved path from topology configuration
- **Agent:** Recommended agent matching the target

**Task Format with Target:**

```markdown
## Task 3: Create User Login API

**Target:** backend
**Working Directory:** packages/api
**Agent:** ring:backend-engineer-golang

**Files to Create/Modify:**
- `packages/api/internal/handlers/auth.go`
- `packages/api/internal/services/auth_service.go`

...rest of task...
```

**Target Assignment Rules:**

| Target | When | Agent |
|--------|------|-------|
| `backend` | API endpoints, services, data layer, CLI | `ring:backend-engineer-{golang,typescript}` |
| `frontend` | UI components, pages, BFF routes | See [Frontend Tasks (api_pattern aware)](#frontend-tasks-api_pattern-aware) |
| `shared` | CI/CD, configs, docs, cross-module | `ring:devops-engineer` or `ring:general-purpose` |

**Working Directory Resolution:**

| Topology Structure | Backend Path | Frontend Path |
|-------------------|--------------|---------------|
| `single-repo` | `.` | `.` |
| `monorepo` | `topology.modules.backend.path` | `topology.modules.frontend.path` |
| `multi-repo` | `topology.modules.backend.path` (absolute) | `topology.modules.frontend.path` (absolute) |

## Agent Selection

### Backend Tasks

| Task Type | Agent |
|-----------|-------|
| Go backend API/services | `ring:backend-engineer-golang` |
| TypeScript backend API/services | `ring:backend-engineer-typescript` |

### Frontend Tasks (api_pattern aware)

**Read `api_pattern` from topology configuration to determine correct agent:**

| API Pattern | Task Type | Agent |
|-------------|-----------|-------|
| `direct` | UI components, pages, forms | `ring:frontend-engineer` |
| `direct` | Server Actions, data fetching | `ring:frontend-engineer` |
| `direct` | Server Components with data loading | `ring:frontend-engineer` |
| `bff` | API routes (`/api/*`) | `ring:frontend-bff-engineer-typescript` |
| `bff` | Data aggregation, transformation | `ring:frontend-bff-engineer-type

... [truncated for markdown readability]

## Skill2Query (before)

- Use ring:writing-plans to Creates comprehensive implementation plans with exact file paths, complete code examples, and verification steps for engineers with zero codebase context
- Can you Creates comprehensive implementation plans with exact file paths, complete code examples, and verification steps for engineers with zero codebase context?
- I need help with Creates comprehensive implementation plans with exact file paths, complete code examples, and verification steps for engineers with zero codebase context
