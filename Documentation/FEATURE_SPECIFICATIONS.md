# FEATURE SPECIFICATIONS

This document defines modular architecture targets for custom gameplay systems.

---

## 1) Nuzlocke Mode Manager

### Purpose
Provide an optional hardcore ruleset with configurable restrictions.

### Responsibilities
- Enable/disable Nuzlocke mode.
- Enforce one encounter per area/route.
- Handle fainted creature restrictions (release/box lock/marking).
- Expose helper checks to battles and capture flow.

### Inputs / Outputs
- **Input:** battle outcomes, encounter events, player settings.
- **Output:** rule decisions (allowed/blocked), player messages, state updates.

### Integration Points
- Encounter system hooks
- Battle result hooks
- Party/box management hooks

---

## 2) Region Progression Manager

### Purpose
Centralize story and progression gating across maps.

### Responsibilities
- Track key progression milestones (badges, events, flags).
- Check whether player can enter specific maps/areas.
- Return friendly lock messages when blocked.
- Support debug bypass during development.

### Inputs / Outputs
- **Input:** current player flags, destination map id, event state.
- **Output:** access decision, reason text, updated progression state.

### Integration Points
- Map transfer events
- Story cutscene completion flags
- Gym badge awards

---

## 3) Encounter Tracking System

### Purpose
Record and expose encounter history per route/map.

### Responsibilities
- Track first encounter by area.
- Record species encountered for balancing/debug.
- Provide query methods for Nuzlocke and progression systems.
- Persist data safely in save data.

### Inputs / Outputs
- **Input:** map id, species id, encounter type.
- **Output:** updated encounter ledger, route encounter status.

### Integration Points
- Wild encounter start/end events
- Capture success events
- Nuzlocke enforcement checks

---

## 4) Trade Evolution Device

### Purpose
Offer an in-world alternative to link trades for specific evolutions.

### Responsibilities
- Validate if creature species can use device.
- Trigger trade-evolution flag logic without real trade.
- Enforce balancing rules (cost, item requirement, cooldown, unlock milestone).
- Provide clear player feedback for success/failure.

### Inputs / Outputs
- **Input:** selected party member, progression status, required items/currency.
- **Output:** evolution trigger result, updated inventory/state.

### Integration Points
- Party menu or dedicated NPC/event interaction
- Evolution logic pipeline
- Region progression unlock checks

---

## Shared Architectural Rules

1. Keep each system in separate script files/modules.
2. Include a `Config` section in each manager for easy tuning.
3. Expose small public methods with descriptive names.
4. Avoid hardcoding map IDs/species IDs directly in logic; move to config constants.
5. Add beginner-friendly comments above non-obvious logic.
6. Build systems so they can be disabled independently during testing.
