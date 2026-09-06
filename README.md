# From GPT-2 to Kimi3

An interactive, single-file research-explainer website that traces how language-model architecture evolved from GPT-2 to Kimi K3.

The site is designed as a visual narrative rather than a static blog post: each major idea is presented as an interactive diagram, live mini-simulation, or mathematical explainer. The goal is to help readers understand not just what changed, but why those changes became necessary as models scaled in depth, sequence length, memory pressure, and routing complexity.

## Project Overview

This project explains the architecture journey from:

- GPT-2 and decoder-only Transformers
- softmax attention and its quadratic cost
- KV cache growth during autoregressive decoding
- linear attention and re-association
- DeltaNet and gated forgetting
- Kimi Delta Attention (KDA)
- hybrid KDA + MLA stacks
- Multi-head Latent Attention (MLA)
- Mixture of Experts (MoE)
- Attention Residuals (AttnRes)
- the assembled Kimi K3 architecture

The website is intentionally implemented as a single `index.html` file so it is easy to:

- read end-to-end
- deploy as a static site
- host on Vercel or GitHub Pages
- fork and customize without a build system

## Why This Project Exists

Most writeups about modern model architectures mention terms like KDA, MLA, MoE, or AttnRes in isolation. This project aims to present them as a connected engineering story.

The central thesis of the site is:

1. GPT-style attention gives strong retrieval but becomes expensive.
2. KV cache solves repeated recomputation but introduces memory growth.
3. Linear/recurrent memory reduces decode cost but must compress information.
4. KDA improves recurrent memory with fine-grained gating.
5. MLA preserves token-level retrieval with a smaller cache.
6. MoE scales capacity without activating all parameters per token.
7. AttnRes helps depth scale more cleanly by retrieving over earlier representations.
8. Kimi K3 combines these ideas into one production-scale design.

## What The Site Contains

The page is structured as a long-form interactive walkthrough. Each section includes explanatory prose and one or more figures.

### Section Breakdown

`00 / Abstract`
- Sets up the core argument of the page.

`01 / Foundations`
- GPT-2 forward pass
- Transformer block expansion
- technical/intuitive mode switching

`02 / Foundations`
- causal attention matrix
- token-by-token inspection
- live scaled dot-product attention equation

`03 / Foundations`
- quadratic cost visualization
- separate treatment of training, prefill, decode, and KV-cache memory

`04 / Foundations`
- KV cache walkthrough
- prefill vs decode stepping

`05 / Linear Memory`
- linear attention re-association
- kernel intuition

`06 / Linear Memory`
- DeltaNet state update with live numbers

`07 / Linear Memory`
- Gated DeltaNet and forgetting behavior

`08 / Kimi's Attention`
- KDA pathways
- intuition/research toggle
- pan/zoom exploration

`09 / Kimi's Attention`
- hybrid stack composer
- KDA vs MLA memory tradeoffs

`10 / Kimi's Attention`
- MLA latent compression walkthrough

`11 / Scaling & Depth`
- Mixture of Experts routing demo

`12 / Scaling & Depth`
- Attention Residuals vs standard residual accumulation

`13 / Scaling & Depth`
- Kimi K3 architecture map
- clickable component detail panel

`14 / Synthesis`
- evolution map across all mechanisms

`15 / Synthesis`
- complexity explorer
- chunk-size tradeoff view

`16 / Synthesis`
- prefill vs decode mechanism viewer

`17 / Synthesis`
- mathematical playground

`18 / Comparisons`
- final comparison surface for model-mechanism tradeoffs

## Core Features

- Single-file static website in `index.html`
- Dark research-style visual design
- Custom SVG favicon in `favicon.svg`
- Interactive SVG diagrams
- Live mathematical rendering using KaTeX
- Sliders, toggles, zoom controls, presets, and step-through states
- Clickable architecture map with contextual explanation panel
- Mobile/tablet responsive behavior
- Deployable without any framework or build pipeline

## Tech Stack

This project is intentionally lightweight.

- `HTML` for structure
- `CSS` for styling and layout
- `Vanilla JavaScript` for all interactivity and rendering
- `SVG` for diagrams and dynamic figures
- `KaTeX` via CDN for mathematical typesetting
- Google Fonts via CDN for typography

There is no bundler, no framework, and no backend.

## Repository Structure

```text
.
├── index.html
├── favicon.svg
└── README.md
```

## Running Locally

Because this is a static site, the simplest option is to open `index.html` directly in a browser.

That said, using a local static server is usually better for testing:

### Option 1: Python

```bash
python3 -m http.server 8000
```

Then open:

[http://localhost:8000](http://localhost:8000)

### Option 2: VS Code / Cursor Live Server

If you use a live preview extension or built-in static preview, open the project folder and serve `index.html`.

## Deployment

This site is static, so deployment is straightforward.

### Vercel

Recommended if you already use it often.

1. Push the project to GitHub.
2. Import the repo into Vercel.
3. No framework preset is required.
4. Vercel should detect and serve `index.html` automatically.
5. Deploy.

### GitHub Pages

1. Push the project to a GitHub repository.
2. Open the repository on GitHub.
3. Go to `Settings` -> `Pages`.
4. Under `Build and deployment`, choose:
   - `Source`: `Deploy from a branch`
   - `Branch`: `main`
   - `Folder`: `/ (root)`
5. Save.
6. Wait for GitHub to publish the site.

## Publishing Notes

Before publishing, check the following:

- all interactive controls work on desktop and mobile
- KaTeX renders correctly after deployment
- external CDN resources load successfully
- section jumps and clickable maps behave correctly
- asset links stay relative, such as `./favicon.svg`

Because this project uses CDN-hosted fonts and KaTeX, the client browser needs normal internet access for those external resources.

## Content Philosophy

This page mixes:

- `SOURCE` claims: facts drawn from papers, writeups, or reported architecture details
- `ILLUSTRATIVE` explanations: simplified demos or toy examples used only for intuition

That distinction is deliberately visible inside the website so readers can tell what is directly sourced versus what is a teaching abstraction.

## Customization Guide

You may want to adapt this site into a personal technical article or portfolio piece. Common edits include:

### 1. Title and branding

Update:

- `<title>` in `index.html`
- hero heading
- sidebar brand text
- footer credit

### 2. References

Update the references and article links in this `README.md` and in the page content.

### 3. Figures

Each figure is rendered directly in JavaScript inside `index.html`. If you want to:

- restyle nodes
- change figure labels
- adjust dimensions
- tweak responsive layout
- revise formulas or captions

you can do so in the corresponding section logic in the same file.

### 4. Colors

The design system lives near the top of `index.html` in the `:root` CSS variables, including:

- `--bg`
- `--panel`
- `--ink`
- `--attn`
- `--mem`
- `--route`
- `--latent`
- `--res`

### 5. Add your own writeup

Use the placeholder section below to attach your own article once it is published.

## Article Links

### Reference Article / Post Used During Development

The external link that informed this project during development:

- [waterloo_intern post on X](https://x.com/waterloo_intern/status/2081762065392541951)

### Kimi Linear Paper Mentioned In The Site

- [Kimi Linear, arXiv:2510.26692](https://arxiv.org/abs/2510.26692)

### Your Article Link

Add your own published article link here:

- [From GPT-2 to Kimi K3: The Journey of 22,580 Models in One](https://chatteronai.hashnode.dev/from-gpt-2-to-kimi-k3-the-journey-of-22-580-models-in-one?utm_source=hashnode&utm_medium=feed)

Example:

- `[From GPT-2 to Kimi3: A Visual Architecture Breakdown](https://your-site.com/your-article)`

## Suggested README Update After You Publish

Your article link is now added above. You can still optionally add:

- where the article is published
- date of publication
- whether the article and the interactive site are identical or complementary

## Known Characteristics

This site is:

- static
- CDN-dependent for fonts and KaTeX
- optimized for modern browsers
- best experienced on desktop, but responsive on smaller screens

It is not:

- a backend application
- a framework app
- a paper reproduction in the academic sense
- a benchmark harness

## Engineering Notes

- The site uses SVG extensively for diagrams because SVG gives precise control over geometry, labels, colors, and interaction.
- Mathematical expressions are rendered at runtime via KaTeX.
- Many figures are intentionally stateful and educational rather than purely decorative.
- The architecture map and figure controls were reviewed for interaction edge cases such as toggles, panning, zooming, click targets, and responsive overflow.

## Who This Is For

This project is useful for:

- ML engineers
- software engineers learning model systems
- students studying attention architectures
- technical writers creating model explainers
- anyone wanting a visual summary of how Kimi K3 differs from earlier designs

## Credits

Website credit in the page footer:

`Made with ♥ by Ujjwal Balaji`

## License

Add your preferred license here if you plan to make the repository public.

Common choices:

- MIT
- Apache-2.0
- All Rights Reserved

Example placeholder:

`License: TBD by project author`
