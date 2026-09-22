// Configuration follows https://zensical.org/docs/authoring/math/.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
};
