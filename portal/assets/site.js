// xtbflow 项目主页：悬停提示与数据目录筛选。
// Material 的 instant 导航未开启，页面每次完整加载，这里只需在 DOMContentLoaded 时绑定一次。
(function () {
  "use strict";

  // 悬停/聚焦提示：任何带 data-tip 的元素。数值同时出现在表格或文字里，提示只是便利。
  var tip = document.createElement("div");
  tip.className = "xf-tip";
  tip.setAttribute("role", "tooltip");

  function show(el, x, y) {
    tip.textContent = el.getAttribute("data-tip");
    tip.classList.add("xf-on");
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(x + 14, window.innerWidth - w - 8);
    var top = y + 16 + h > window.innerHeight ? y - h - 12 : y + 16;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function hide() { tip.classList.remove("xf-on"); }

  function bind() {
    document.body.appendChild(tip);
    document.querySelectorAll("[data-tip]").forEach(function (el) {
      el.addEventListener("mousemove", function (ev) { show(el, ev.clientX, ev.clientY); });
      el.addEventListener("mouseleave", hide);
      el.addEventListener("focus", function () {
        var r = el.getBoundingClientRect();
        show(el, r.left + r.width / 2, r.top + r.height / 2);
      });
      el.addEventListener("blur", hide);
    });

    // 数据目录筛选：按整行文字（含路径和 PR 号）匹配，空格分隔的多个词须全部命中。
    document.querySelectorAll("input[data-filter-target]").forEach(function (input) {
      var table = document.getElementById(input.getAttribute("data-filter-target"));
      if (!table) return;
      input.addEventListener("input", function () {
        var terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
        table.querySelectorAll("tbody tr").forEach(function (row) {
          var text = row.textContent.toLowerCase();
          row.style.display = terms.every(function (t) { return text.indexOf(t) !== -1; }) ? "" : "none";
        });
      });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
