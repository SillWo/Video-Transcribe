# Railway — Style Reference
> dusk sky over a departure board. Build calm purple-black station halls where serif headlines arrive softly, one violet button boards the action and each chapter rides a tinted carriage.

**Theme:** dark

Source measurements are normalized; roles and recommendations are interpreted. Font summary lists are independent, not paired by position. HTML examples are reconstructions, not source components.

Railway is a night-time train station rendered as a cloud console. A purple-black #13111c canvas holds a sequence of 16px-rounded chapter panels, each tinted its own dusk tone (#1b2132 navy, #1f2b2e pine, #251823 plum), while a line-drawn train rides a vertical rail beside the copy. IBM Plex Serif headlines at 400–500 with negative tracking give the page a quiet, literary calm; Inter carries everything functional at 14–20px. Colour is rationed: one #553f83 violet Deploy button, lilac #a78bfa highlights, small green and peach chapter tags, and a single magenta radial All Aboard pill in the closing train doorway. Proof comes from real console windows, a live split-flap stats board and a painted starfield hero.

## Colors

| Name | Value | Role |
|------|-------|------|
| Midnight Canvas | `#13111c` | Page background, top bar and untinted chapters — a purple-tinged black that keeps the whole page at dusk |
| Platform White | `#f7f7f8` | All text from nav to serif headlines and button labels (headings compute #ffffff, rounded here) |
| Mist Gray | `#a1a0ab` | Lead paragraphs, feature descriptions, handles, secondary labels and the boarding subline |
| Deploy Violet | `#553f83` | The hero Deploy button fill — a muted, dusky violet that reads as twilight rather than a neon call to action |
| Lilac Signal | `#a78bfa` | Inline highlight phrases in footer feature cards and the active dock tab label — the brighter echo of Deploy Violet |
| Night Navy | `#1b2132` | Build-and-deploy chapter panel — the cool blue carriage that follows the hero |
| Pine Slate | `#1f2b2e` | Monitor-and-observe chapter panel, a green-grey tint behind the metrics console |
| Plum Dusk | `#251823` | Customer trust chapter behind the quote cards and the tweet marquee |
| Console Black | `#0b0b0f` | Product console windows and split-flap tiles, the inset darkest layer below every chapter tint |
| Track Gray | `#6b7280` | Unlit split-flap label letters such as USERS and LOGS, the dimmest text on the page |
| Signal Green | `#42946e` | Network and Evolve chapter tags, the matching timeline rail, Online checks and the monospace status line |
| Ember Peach | `#eab38a` | Scale chapter tag text and its rail — a warm lamp-light cue against the cold dark panels |
| Boarding Glow | `radial-gradient(73.46% 138.39% at 50.21% 0%, #aa0aaa 50%, #6d1dbd 75.47%, #381dbd 100%)` | The All Aboard button only — magenta-to-indigo radial fill, the one saturated moment at the end of the journey |

## Typography

### IBM Plex Serif — The display voice: 54px/500 for the hero, 36px/400 for chapter headings and 40px/400 for the trust and stats headings. A bookish serif at gentle weights with -1.96px to -0.72px tracking makes infrastructure copy feel calm and editorial, the opposite of the bold grotesk most cloud sites shout in.
- **Substitute:** Georgia, Cambria, Times New Roman, serif
- **Licence:** open (SIL Open Font License); free to use and self-host
- **Weights:** 400, 500
- **Sizes:** 36px, 40px, 54px
- **Line height:** 1.12, 1.20, 1.33
- **Letter spacing:** -1.96px at 54px; -0.8px at 40px; -0.72px at 36px

### Inter — Everything functional: 14px/500 navigation and tags, 16px/400 body, 16px/600 feature titles, 18px/500 leads at 1.6 line-height and 20px/500 hero buttons. Tracking stays at 0 except slight negatives on the larger sizes, so the sans recedes and lets the serif carry the personality.
- **Substitute:** -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif
- **Licence:** open (SIL Open Font License); free to use and self-host
- **Weights:** 400, 500, 600, 700
- **Sizes:** 12px, 14px, 16px, 18px, 20px
- **Line height:** 1.25, 1.40, 1.43, 1.50, 1.56, 1.60, 1.75
- **Letter spacing:** -0.2px at 20px; -0.09px at 18px; -0.1px at 16px quotes; -0.07px at 14px tags; otherwise 0

### ui-monospace — The departure-board face: 40px/400 uppercase glyphs at 56px line-height, one per split-flap tile, plus the 12px All systems operational status line. A platform face, it renders as SF Mono on macOS and Consolas on Windows (slashed zeros in the capture); JetBrains Mono is the closest public substitute.
- **Substitute:** SFMono-Regular, SF Mono, Consolas, Liberation Mono, Menlo, monospace
- **Weights:** 400
- **Sizes:** 12px, 40px
- **Line height:** 1.40, 1.50
- **Letter spacing:** normal

### Inter Tight — The second sans voice, used twice: 32px/400 for ...and loved by developers above the tweet marquee and 40px/600 at -0.6px tracking for the closing A better future is now boarding. The narrower, heavier cut makes the finale feel like a station sign rather than another serif chapter.
- **Substitute:** Inter, sans-serif
- **Licence:** open (SIL Open Font License); free to use and self-host
- **Weights:** 400, 600
- **Sizes:** 32px, 40px
- **Line height:** 1.20, 1.38
- **Letter spacing:** -0.6px at 40px; normal at 32px

### Type Scale

| Role | Family | Weight | Size | Line Height | Letter Spacing |
|------|--------|--------|------|-------------|----------------|
| body-small | Inter | 400 | 14px | 1.43 | 0px |
| nav | Inter | 500 | 14px | 1.43 | 0px |
| body | Inter | 400 | 16px | 1.5 | 0px |
| feature-title | Inter | 600 | 16px | 1.5 | 0px |
| lead | Inter | 500 | 18px | 1.6 | -0.09px |
| button-large | Inter | 500 | 20px | 1.6 | -0.2px |
| sub-display | Inter Tight | 400 | 32px | 1.38 | 0px |
| section-heading | IBM Plex Serif | 400 | 36px | 1.33 | -0.72px |
| split-flap | ui-monospace | 400 | 40px | 1.4 | 0px |
| trust-heading | IBM Plex Serif | 400 | 40px | 1.2 | -0.8px |
| boarding-heading | Inter Tight | 600 | 40px | 1.2 | -0.6px |
| hero-display | IBM Plex Serif | 500 | 54px | 1.12 | -1.96px |

## Spacing & Layout

**Base unit:** 4px

**Density:** comfortable

- **Page max-width:** 1160px
- **Section gap:** 96px
- **Card padding:** 24px
- **Element gap:** 16px

### Border Radius

- **flaps:** 4px
- **ghost-buttons:** 6px
- **buttons:** 8px
- **tags:** 8px
- **windows:** 10px
- **cards:** 12px
- **panels:** 16px
- **boarding-pill:** 28px
- **avatars:** 9999px

## Components

### Station Top Bar
**Role:** Top-level site navigation

A 72px Midnight Canvas #13111c bar with the roundel wordmark at left. Centre the dropdown triggers as transparent 16px/400 Inter Platform White #f7f7f8 buttons with 6px 12px padding and 6px radius; Sign in and Book a demo sit at right as 14px/500 ghost links, 40px high with 0 12px padding. No border, no shadow.

### Dusk Hero Panel
**Role:** Opening statement over painted sky

A full-width panel with 16px corners inset 16px from the viewport, filled with the painted dusk-sky artwork. Centre an IBM Plex Serif 54px/500 Platform White #f7f7f8 headline at 60.48px line-height and -1.96px tracking, a 20px/400 Inter subtitle at 28px line-height, the button pair, and a console window rising from the panel bottom.

### Deploy Violet Button
**Role:** Primary hero action

Fill with Deploy Violet #553f83, Platform White #f7f7f8 label in Inter 20px/500 at 32px line-height and -0.2px tracking, 12px 24px padding, 8px radius and a 1px rgba(255, 255, 255, 0.15) border, 58px high. Trailing arrow in the label; no shadow, no gradient.

### Smoked Glass Button
**Role:** Secondary hero action

Same geometry as the Deploy Violet Button — 58px high, 8px radius, 12px 24px padding, Inter 20px/500 Platform White #f7f7f8 label — but filled with rgba(0, 0, 0, 0.35) and outlined in 1px rgba(255, 255, 255, 0.25) so the painted sky shows through.

### Frosted Ghost Button
**Role:** Tertiary section action

A rgba(255, 255, 255, 0.1) fill with Platform White #f7f7f8 Inter 16px/400 text at 24px line-height, 10px 16px padding, 6px radius and no border, 44px high. Used once per chapter, e.g. Read customer stories → under the trust heading.

### Chapter Tag Pill
**Role:** Coloured chapter label above a heading

An 8px-radius label in Inter 14px/500 at 20px line-height and -0.07px tracking with 6px 12px padding. Neutral chapters use a rgba(255, 255, 255, 0.1) fill with Platform White #f7f7f8 text; network uses Signal Green #42946e text on rgba(66, 148, 110, 0.1), scale uses Ember Peach #eab38a on rgba(237, 133, 47, 0.12).

### Track Rail Timeline
**Role:** Vertical progress line beside each chapter

A thin vertical rail running down the left of the 1160px column with a small outline train at its head and a hollow ring node level with the chapter tag. Tint the rail and ring with the chapter accent — teal-grey by default, Signal Green #42946e for network, Ember Peach #eab38a for scale.

### Console Window
**Role:** Product UI proof inside a chapter

A Console Black #0b0b0f window with 10px radius and a 1px rgba(255, 255, 255, 0.1) border, about 1160px wide. Give it a breadcrumb header with Architecture, Observability, Logs and Settings tabs, a dotted canvas grid, and Carriage Panel #1c1a28 service nodes with 8px corners and Signal Green #42946e check marks.

### Chapter Switcher Dock
**Role:** Tabbed chapter selector on the hero console

A dark bar overlapping the bottom edge of the hero console holding Deploy, Network, Scale, Monitor and Evolve tabs in Inter 16px/400 with outline icons, separated by 1px rgba(255, 255, 255, 0.1) dividers. The active tab is a violet-tinted 8px-radius fill with Lilac Signal #a78bfa text, 32px high.

### Feature Triad Row
**Role:** Three-column feature summary

Three equal columns under a console, each a small outline icon, a 16px/600 Inter Platform White #f7f7f8 title at 24px line-height and 16px/400 Mist Gray #a1a0ab copy. Close with a 1px rgba(255, 255, 255, 0.1) rule and an Alternative to label followed by 32px monochrome competitor logos.

### Customer Quote Card
**Role:** Enterprise testimonial

A 360px by 500px card with a rgba(255, 255, 255, 0.05) fill on Plum Dusk #251823, 12px radius and 48px padding. Customer wordmark at top, Inter 16px/400 quote at 28px line-height and -0.1px tracking, then a 9999px avatar with a 14px/700 name and 14px/400 Mist Gray #a1a0ab title.

### Tweet Marquee Card
**Role:** Developer social proof in a scrolling row

A 360px-wide rgba(255, 255, 255, 0.05) card with 8px radius and 24px 24px 32px padding, in an edge-to-edge auto-scrolling row. Stack a 9999px avatar, a 14px/700 name and 14px/400 Mist Gray #a1a0ab handle, then the post in Inter 14px/400 at 21px line-height.

### Split-Flap Stats Board
**Role:** Live usage counters as a departure board

A full-bleed grid of 44px-wide Console Black #0b0b0f flaps, each split into 4px 4px 0 0 and 0 0 4px 4px halves with 1px rgba(255, 255, 255, 0.086) seams. Set glyphs in ui-monospace 40px/400 uppercase at 56px line-height: labels in Track Gray #6b7280, live counts in Platform White #f7f7f8, flipping on 0.3s linear transforms.

### Boarding Door Finale
**Role:** Closing call to action inside the train

A full-bleed line-drawn carriage on Midnight Canvas #13111c whose centre doors slide apart to reveal a warm magenta-copper glow. Centre an Inter Tight 40px/600 Platform White #f7f7f8 heading at 48px line-height and -0.6px tracking, a 20px/400 Mist Gray #a1a0ab subline at 28px line-height, then the All Aboard button.

### All Aboard Button
**Role:** Final conversion action

Fill with the Boarding Glow radial gradient (#aa0aaa at 50%, #6d1dbd, #381dbd), a 1px rgb(160, 95, 207) border, 28px radius and 12px 24px padding, 56px high. Platform White #f7f7f8 Inter 20px/500 label at 32px line-height; glow of rgba(180, 40, 180, 0.25) 0 0 6px and rgba(102, 43, 223, 0.25) 0 0 16px.

### Footer Feature Card
**Role:** Announcement links in the footer

A Carriage Panel #1c1a28 card with a 1px rgba(255, 255, 255, 0.1) border, 12px radius and 16px 24px padding. Lead with an Inter 16px/500 title whose second half is a Lilac Signal #a78bfa phrase, then 16px/400 Mist Gray #a1a0ab copy and an underlined Watch the film → link.

### Station Footer
**Role:** Site map, compliance and status

A 1280px footer with 48px 0 96px padding on Midnight Canvas #13111c: two feature cards at left, six link columns at right with bold headings and 16px/400 Platform White #f7f7f8 links at 26px line-height, then a 1px rule, the roundel, and a 12px ui-monospace Signal Green #42946e All systems operational status.

## Do's and Don'ts

### Do
- Use Midnight Canvas #13111c as the page field and step each chapter into a 16px-radius tinted panel (Night Navy #1b2132, Pine Slate #1f2b2e, Plum Dusk #251823) inset 16px from the viewport edge.
- Set headlines in IBM Plex Serif: 54px/500 for the hero at 1.12 line-height and -1.96px tracking, 36px/400 for chapter headings at 48px line-height and -0.72px tracking.
- Reserve Deploy Violet #553f83 for the one primary action: 8px radius, 12px 24px padding, a 20px/500 Inter label with -0.2px tracking and a 1px rgba(255, 255, 255, 0.15) border.
- Keep body and UI text in Inter at 14–18px, with 18px/500 leads in Mist Gray #a1a0ab ending in a Platform White #f7f7f8 Learn more → link.
- Apply chapter colour only to small markers: the 8px-radius tag pill, the timeline rail and its ring node, using Signal Green #42946e for network and Ember Peach #eab38a for scale.
- Build product proof as real console windows on Console Black #0b0b0f with a 10px radius, a 1px rgba(255, 255, 255, 0.1) border and a dotted canvas grid.
- Keep ui-monospace for the split-flap board alone: 40px/400 uppercase at 56px line-height on Console Black #0b0b0f tiles with 4px corners, plus the 12px status line.

### Don't
- Do not set headlines in Inter or in bold serif weights; display type stays IBM Plex Serif 400–500 with negative tracking.
- Do not use pure black or a neutral gray canvas; the field is the purple-tinged Midnight Canvas #13111c.
- Do not round the hero buttons into pills; primary and smoked-glass actions use 8px corners, frosted ghost buttons 6px, and only the closing All Aboard button goes to 28px.
- Do not spread Deploy Violet #553f83 or the Boarding Glow gradient across surfaces or headings; each fills one button, and Lilac Signal #a78bfa stays on inline phrases and active tabs.
- Do not float cards on blurred drop shadows; separate surfaces with 1px rgba(255, 255, 255, 0.1) borders, inset rings and tonal steps.
- Do not replace the painted dusk sky and line-drawn train art with stock photography or abstract gradient blobs.
- Do not use Signal Green #42946e or Ember Peach #eab38a as button or panel fills; they tint 0.1–0.12 alpha tag backgrounds, rails and status icons only.

## Elevation

Depth comes from tone, not lift. Chapter panels sit one step lighter than Midnight Canvas with a 1px rgba(255, 255, 255, 0.1) inset ring; console windows drop darker to Console Black; cards use 0.05-alpha white fills and 1px 0.1-alpha borders. Only two long, very soft five-layer shadows exist on large demo blocks, plus a tight magenta 6px/16px glow that is the All Aboard button's only lift.

## Surfaces

- **Midnight Canvas** (`#13111c`) — Page field, top bar, untinted chapters and the footer
- **Night Navy** (`#1b2132`) — Build-and-deploy chapter panel
- **Pine Slate** (`#1f2b2e`) — Monitor-and-observe chapter panel
- **Plum Dusk** (`#251823`) — Customer trust chapter and testimonial field
- **Carriage Panel** (`#1c1a28`) — Footer feature cards and raised service nodes
- **Console Black** (`#0b0b0f`) — Inset product consoles and split-flap tiles

## Imagery

Two visual registers alternate. The hero is a hand-painted, anime-like dusk sky: starfield, layered mauve and peach clouds and later a mountain line and passing train, all in a muted violet-teal grade cropped inside the 16px panel. Everything else is line-drawn station hardware or real product UI — a tiny outline train riding the timeline rail, a station platform with tactile paving, a train carriage with rounded windows and sliding doors, and a split-flap departure board ticking live counts. Product proof is the actual Railway canvas: service nodes, metric charts and environment stacks in dark console windows with a dotted grid. Icons are thin monochrome outline marks, chapter icons pick up the chapter accent, and customer logos and portraits appear in white or circular 9999px crops. There is no stock photography.

## Layout

The page is a vertical train journey. A 72px top bar holds the wordmark, centred dropdown navigation and two ghost links; below it each major block is a full-width panel with 16px corners inset 16px from the viewport, so the canvas shows as a thin frame between carriages. The hero is centred: serif headline, subtitle, a Deploy and Agents pair, then a 1060px console window rising out of a painted sky with a five-tab chapter dock on its lower edge. Feature chapters share one template in a 1160px column: a left rail with a train and ring node, a tag pill, a two-line serif heading, an 18px lead with Learn more, then a full-width console or a split copy-and-visual row, a three-column feature row and an Alternative to logo strip. Vertical padding is generous (96–128px); content inside stays compact. Trust centres three quote cards over a tweet marquee; then a full-bleed split-flap board, a train whose doors open on the Inter Tight boarding CTA, and the footer.

## Similar Brands

- **Fly.io** — Shares an illustrated, storybook art direction wrapped around a developer deployment platform instead of abstract tech graphics.
- **Linear** — Uses near-black canvases, 1px low-alpha white borders and real product UI as the main visual proof.
- **Raycast** — Same habit of framing the product as dark, dense application windows set on a darker page field.
- **Supabase** — Shares dark developer-platform pages where actual dashboard panels and compact tag labels carry each feature section.
