#set page(
  paper: "a4",
  margin: (x: 1.25cm, y: 1.15cm),
  numbering: "1",
)
#set text(font: "Libertinus Serif", size: 9pt, lang: "en")
#set par(justify: true, leading: 0.48em, spacing: 0.55em)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(
  above: 0em,
  below: 0.45em,
  align(center, text(size: 15pt, weight: "bold", it.body)),
)
#show heading.where(level: 2): it => block(
  above: 0.65em,
  below: 0.2em,
  text(size: 10.2pt, weight: "bold", it.body),
)
#set table(inset: 2.2pt, stroke: 0.35pt + luma(160))
#show table: set text(size: 7.8pt, hyphenate: false)
#show figure: set block(above: 0.35em, below: 0.35em)
#show raw: set text(font: "DejaVu Sans Mono", size: 7.2pt)
#show link: set text(fill: rgb("#1a4f8a"))
#set list(indent: 0.9em, spacing: 0.35em)
#set enum(indent: 0.9em, spacing: 0.35em)
