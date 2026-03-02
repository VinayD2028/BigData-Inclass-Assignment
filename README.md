# pokemon-world-chronicles

A beginner-friendly, GitHub-ready repository scaffold for a **2D monster-catching RPG** built with **Pokémon Essentials** and **RPG Maker XP**.

This project is structured to support long-term development with modular custom systems, clear documentation, and safe version control practices.

---

## Project Vision

`pokemon-world-chronicles` aims to provide:
- A clean foundation for story, maps, and gameplay systems.
- Modular architecture for custom mechanics.
- Documentation-first development for easier collaboration.
- Compatibility with Pokémon Essentials workflows.

---

## Repository Structure

```text
pokemon-world-chronicles/
├── Audio/                  # Music, SE, ME, BGS, BGM assets
├── Data/                   # Core game data generated/used by Essentials
├── Documentation/          # Design docs, specs, and planning files
├── Graphics/               # Tiles, sprites, UI, animations
├── Maps/                   # Optional map exports/notes and map planning docs
├── Plugins/                # Future Essentials plugin modules
├── Scripts/                # Custom Ruby systems and managers
├── .gitignore              # RMXP + Essentials friendly ignore rules
└── README.md               # Main onboarding guide
```

---

## Modular Custom Systems

This scaffold includes starter script templates for:

1. **Nuzlocke Mode Manager** (`Scripts/Nuzlocke_Manager.rb`)
2. **Region Progression Manager** (`Scripts/Region_Manager.rb`)
3. **Encounter Tracking System** (`Scripts/Encounter_Manager.rb`)

And a documented architecture plan for:

4. **Trade Evolution Device** (planned implementation details in docs)

See: `Documentation/FEATURE_SPECIFICATIONS.md`.

---

## Step-by-Step Setup Guide

### 1) Install RPG Maker XP and Pokémon Essentials

1. Install **RPG Maker XP**.
2. Obtain a legal copy of **Pokémon Essentials** from the official community source.
3. Extract Pokémon Essentials into a new local folder named `pokemon-world-chronicles`.
4. Open the project in RPG Maker XP once to ensure all base files are generated correctly.
5. Close RPG Maker XP before adding version control.

> Tip: Keep a clean backup of your fresh Essentials install before heavy modifications.

### 2) Apply this GitHub-ready repository structure

1. Ensure the folders in this repository exist (`Data`, `Scripts`, `Graphics`, `Audio`, `Maps`, `Plugins`, `Documentation`).
2. Copy this scaffold into your Essentials project root.
3. Merge carefully if files already exist.

### 3) Connect the project to GitHub

Run these commands from the project root:

```bash
git init
git add .
git commit -m "chore: initialize pokemon-world-chronicles project scaffold"
git branch -M main
git remote add origin https://github.com/<your-username>/pokemon-world-chronicles.git
git push -u origin main
```

If this repo is already initialized, skip `git init` and only add/commit/push changes.

### 4) Recommended versioning strategy

Use a lightweight semantic style aligned to milestones:

- `v0.1.0` → Initial playable prototype
- `v0.2.0` → Core loop stable (battle/catch/progression)
- `v0.3.0` → Major custom systems integrated
- `v1.0.0` → Full public release

Branch strategy for beginners:
- `main`: always stable/playable.
- `feature/<system-name>`: one feature at a time.
- Open pull requests to merge back into `main`.

### 5) First commit checklist

Before your first serious development commit:

- [ ] Project opens in RPG Maker XP without errors.
- [ ] Pokémon Essentials boots in playtest mode.
- [ ] `.gitignore` excludes temp/cache/build artifacts.
- [ ] Documentation files are present and readable.
- [ ] Base managers exist in `Scripts/`.
- [ ] Commit message clearly describes initialization work.

---

## Next Development Steps

1. Fill in game world and narrative details in `PROJECT_OVERVIEW.md`.
2. Prioritize systems in `DEVELOPMENT_ROADMAP.md`.
3. Implement one manager at a time using script templates.
4. Test each feature in isolation before integration.

---

## Important Notes

- Do **not** commit copyrighted assets you don't own the rights to distribute.
- Keep all custom code modular and documented.
- Commit often with small, clear messages.

