# Pure Deep Reinforcement Learning Scheduling System - TODO List

## Project Status: Phase 1, 1.5, 1.6, and 2 Complete - Ready for Phase 3 (Training)
Last Updated: 2025-07-24
**Objective: Build a pure DRL scheduler that learns everything from experience, like AI learning to play a game**

## Core Principles
- You control the game rules (when to play, what machines exist)
- PPO model focuses only on playing the game (which job on which machine)
- No hardcoded strategies - everything learned through rewards
- Simple interface: Raw data in → Schedule out

## Phase 1: Foundation - Build the Game Environment (Week 1) ✅ COMPLETE

### Step 1: Database Connection Layer ✅
- [x] Create simple MariaDB connector in `/app_2/src/data/`
- [x] Query pending jobs: job_id, family_id, sequence, machine_types, processing_time, lcd_date, is_important
- [x] Query machines: machine_id, machine_type
- [x] No validation, just raw data extraction
