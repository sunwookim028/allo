# Team Collaboration Workflow

## Overview
This fork (`origin`) serves as the **team's collaboration hub**. All development happens here. We only push to upstream (`upstream`) when the work is complete and ready for final PR.

## Setup

### Sunwoo's Fork (This Repo)
- **Remote name**: `origin` → `git@github.com:sunwookim028/allo.git`
- **Role**: Source of truth for team collaboration
- **Main branch**: Team's main development branch

### Upstream Repo
- **Remote name**: `upstream` → `git@github.com:cornell-zhang/allo.git`
- **Role**: Final destination (only when complete)
- **Main branch**: Original repo's main

## Workflow

### For Team Members

#### Initial Setup
```bash
# Clone this fork (team's collaboration repo)
git clone git@github.com:sunwookim028/allo.git
cd allo

# Add upstream remote (for reference, rarely used)
git remote add upstream git@github.com:cornell-zhang/allo.git

# Set main to track this fork
git checkout main
git branch --set-upstream-to=origin/main main
```

#### Daily Workflow
```bash
# Start from team's main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/hls-static-types

# Work, commit, push to remote
git add .
git commit -m "feat: add static type support"
git push origin feature/hls-static-types

# Collaborate: teammates can fetch and review
# git fetch origin feature/hls-static-types
```

#### Team Collaboration
- **All branches pushed to `origin`** (this fork)
- **PRs created within this fork** for team review
- **No interaction with upstream** until work is complete

### Before Final PR to Upstream

```bash
# Sync with upstream (one-time before PR)
git checkout main
git pull upstream main

# Rebase feature branch
git checkout feature/hls-static-types
git rebase main

# Push to this fork
git push origin feature/hls-static-types --force-with-lease

# Create PR: this-fork → upstream
```

## Branch Strategy

- **`main`** on `origin`: Team's main branch (can diverge from upstream)
- **`feature/*`**: Feature branches on `origin` for collaboration
- **Upstream `main`**: Only touched when creating final PR

## Key Points

✅ **This fork is the collaboration hub** - all team work happens here  
✅ **No frequent upstream syncing** - only before final PR  
✅ **PRs within this fork** - for team review and collaboration  
✅ **Upstream PR only at completion** - when work is ready  

## Directory Structure

```
docs/refreshers/              # Refresher materials (create as needed)
├── hls_static_features/
├── type_system/
└── ir_compilation/
```

## Notes

- This is a **valid workflow** - many teams use forks as collaboration hubs
- This fork's `main` can diverge from upstream during development
- Team members work entirely within this fork ecosystem
- Upstream is only for the final contribution

