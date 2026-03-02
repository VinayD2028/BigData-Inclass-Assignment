# PROJECT OVERVIEW

## Project Name
**pokemon-world-chronicles**

## Genre
2D monster-catching RPG (Pokémon-style progression)

## Engine Stack
- **RPG Maker XP**
- **Pokémon Essentials**
- **Ruby scripts** for custom systems

## Development Philosophy
This project follows a **modular and beginner-friendly architecture**:
- Each custom gameplay feature lives in its own manager module.
- Configuration is centralized and easy to tweak.
- New developers can understand code through clear comments and simple naming.

## Core Gameplay Loop
1. Explore routes/towns.
2. Encounter and catch creatures.
3. Battle trainers and progress story gates.
4. Unlock new areas based on badges/events.
5. Grow team strategy through progression systems.

## Target Custom Systems
1. **Nuzlocke Mode Manager**
   - Optional hardcore ruleset toggle.
   - Handles permadeath and catch limitations.

2. **Region Progression Manager**
   - Controls access rules, milestones, and progression gating.

3. **Encounter Tracking System**
   - Tracks first encounters and route encounter states.

4. **Trade Evolution Device**
   - Allows trade evolutions through in-game mechanic/device.

## Repository Documentation Map
- `README.md` → onboarding and setup steps
- `Documentation/DEVELOPMENT_ROADMAP.md` → milestone timeline
- `Documentation/FEATURE_SPECIFICATIONS.md` → system-level requirements and architecture

## Non-Goals (for early versions)
- Online multiplayer/trading features
- Massive content scope before core systems are stable
- Use of unlicensed/copyrighted external assets
