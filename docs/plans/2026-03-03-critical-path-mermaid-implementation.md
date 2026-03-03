# Critical Path Mermaid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the ALU-path Mermaid diagram for readability and rewrite the memory-path Mermaid diagram into syntax-compatible Mermaid that renders reliably across viewers.

**Architecture:** Keep the surrounding narrative and tables unchanged, and only replace the two Mermaid blocks in the detailed design doc. Use simpler node structure, explicit edges, and left-to-right flow to reduce layout noise and avoid renderer-specific syntax issues.

**Tech Stack:** Markdown, Mermaid, Mermaid CLI (`mmdc`)

---

### Task 1: Update the Mermaid diagrams

**Files:**
- Modify: `docs/detailed-design/00-关键路径时序图.md`
- Verify: `mmdc` render of extracted Mermaid blocks

**Step 1: Capture the approved design direction**

Use a cleaner ALU flow with one node per stage and compact annotations. Rewrite the memory-path flow to use explicit fan-out edges instead of grouped `&` links.

**Step 2: Edit the Markdown Mermaid blocks**

Replace the ALU diagram with a stage-by-stage left-to-right flow. Replace the memory-path diagram with a compatible left-to-right flow that separates Subcore, PRT/ldst, routing branches, and writeback.

**Step 3: Verify rendering**

Run `mmdc` on the extracted ALU and memory Mermaid blocks and confirm both render successfully.

**Step 4: Review for doc consistency**

Check that the updated diagrams still match the explanatory bullets and tables immediately below them.
