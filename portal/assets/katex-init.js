// 渲染 pymdownx.arithmatex（generic 模式）输出的 \( \) 与 \[ \] 公式。
document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement !== "function") return;
  document.querySelectorAll(".arithmatex").forEach(function (el) {
    renderMathInElement(el, {
      delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
      ],
      throwOnError: false
    });
  });
});
