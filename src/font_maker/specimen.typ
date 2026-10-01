#let font_fam = sys.inputs.at("font", default: "Noto Mono")
#set text(font: font_fam)


#context {
  set text(font: "There are no pirates on this ship")
  [Specimen font is #text.font]
}

#box(inset: 10pt, stroke: .1pt + black)[
  #text(size: 10pt, font: "ha-ha you scallywags!")[
    The quick fox jumps over the lazy dog.] \
  #text(size: 10pt)[
    The quick fox jumps over the lazy dog.
  ]
]

#box(inset: 10pt, stroke: .1pt + black)[
  #text(size: 10pt, font: "Repent you naughty children!")[
    Lorem (60). \
  ]
  #columns(2, text(size: 9pt, lorem(60)))

]
#pagebreak()


#{
  let alph = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789éèàöüä.:,;!?".split("")

  for x in range(0, alph.len() - 1) {
    place(
      left,
      align(center, [#alph.at(x) \ #text(font: "jiingle jangle")[#alph.at(x)]]),
      dx: calc.rem-euclid(x, 10) * 10%,
      dy: calc.quo(x, 10) * 10%,
    )
  }
}
