# DEVELOPMENT ROADMAP

This roadmap is designed for beginner-friendly progress with clear milestones.

## Milestone 0 - Repository & Tooling Setup
- [x] Initialize project structure.
- [x] Add `.gitignore` for RMXP + Essentials.
- [x] Create foundational documentation.
- [x] Add manager script templates.

## Milestone 1 - Core Playable Slice
- [ ] Create starter town + 1 route + 1 dungeon.
- [ ] Ensure encounter tables are working.
- [ ] Add first rival/trainer battle.
- [ ] Confirm save/load reliability.

## Milestone 2 - Nuzlocke Rules Integration
- [ ] Implement mode toggle in `Nuzlocke_Manager`.
- [ ] Enforce one encounter per route logic.
- [ ] Enforce fainted party member lockout/release flow.
- [ ] Add player-facing status UI/message hooks.

## Milestone 3 - Region Progression Controls
- [ ] Implement badge/event based gate checks.
- [ ] Connect map transitions to progression manager.
- [ ] Add fallback messages when area is locked.
- [ ] Add debug override for testing progression.

## Milestone 4 - Encounter Tracking Expansion
- [ ] Persist encounter state by map/area.
- [ ] Expose query helpers for other systems.
- [ ] Add analytics hooks (optional) for balance tuning.

## Milestone 5 - Trade Evolution Device
- [ ] Design item/NPC/device interaction flow.
- [ ] Define eligible species list and restrictions.
- [ ] Implement evolution trigger without direct trade.
- [ ] Add balancing constraints (cost, cooldown, progression lock).

## Milestone 6 - Content Scaling
- [ ] Expand routes, towns, and side quests.
- [ ] Add progression pacing checks.
- [ ] Playtest and rebalance encounters/trainers.

## Versioning Targets
- **v0.1.0**: Milestones 0-1 complete.
- **v0.2.0**: Milestones 2-3 complete.
- **v0.3.0**: Milestones 4-5 complete.
- **v1.0.0**: Full narrative/content pass + polish.
