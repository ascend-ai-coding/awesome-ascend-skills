const data = window.SKILLS_APP_DATA || { skills: [], bundles: [], stats: {}, repo: "ascend-ai-coding/awesome-ascend-skills" };
if ("scrollRestoration" in history) {
  history.scrollRestoration = "manual";
}

const state = {
  query: "",
  filter: "all",
  selected: new Set(),
  activeSkillName: "",
};

const filterOptions = [
  { id: "all", label: "全部来源", type: "all", value: "all" },
  { id: "base", label: "基础", type: "category", value: "base" },
  { id: "inference", label: "推理", type: "category", value: "inference" },
  { id: "training", label: "训练", type: "category", value: "training" },
  { id: "profiling", label: "Profiling", type: "category", value: "profiling" },
  { id: "ops", label: "算子", type: "category", value: "ops" },
  { id: "ai-for-science", label: "AI-For-Science", type: "category", value: "ai-for-science" },
  { id: "mindstudio", label: "MindStudio", type: "source", value: "mindstudio" },
  { id: "gitcode-ascend", label: "Gitcode-Ascend", type: "source", value: "gitcode-ascend" },
];

const categoryLabels = {
  base: "基础",
  inference: "推理",
  training: "训练",
  profiling: "Profiling",
  ops: "算子",
  "agent-tools": "工程工具",
  "ai-for-science": "AI for Science",
  bundle: "安装包",
  external: "外部",
};

const els = {
  statsLine: document.querySelector("#statsLine"),
  searchInput: document.querySelector("#searchInput"),
  categoryFilters: document.querySelector("#categoryFilters"),
  skillList: document.querySelector("#skillList"),
  skillDetail: document.querySelector("#skillDetail"),
  resultCount: document.querySelector("#resultCount"),
  selectedCount: document.querySelector("#selectedCount"),
  selectAllVisibleButton: document.querySelector("#selectAllVisibleButton"),
  clearVisibleButton: document.querySelector("#clearVisibleButton"),
  clearSelectionButton: document.querySelector("#clearSelectionButton"),
  bulkNpx: document.querySelector("#bulkNpx"),
  bulkSkillsSh: document.querySelector("#bulkSkillsSh"),
  bulkAgentPrompt: document.querySelector("#bulkAgentPrompt"),
  bundleGrid: document.querySelector("#bundleGrid"),
  toast: document.querySelector("#toast"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split("\n").slice(0, 520);
  let html = "";
  let inCode = false;
  let inMath = false;
  let codeBuffer = [];
  let mathBuffer = [];
  let tableBuffer = [];
  let inList = false;
  let listType = "ul";

  const closeList = () => {
    if (inList) {
      html += `</${listType}>`;
      inList = false;
    }
  };

  const flushCode = () => {
    html += `<pre><code>${escapeHtml(codeBuffer.join("\n"))}</code></pre>`;
    codeBuffer = [];
  };

  const flushMath = () => {
    html += `<div class="math-block">\\[${escapeHtml(mathBuffer.join("\n"))}\\]</div>`;
    mathBuffer = [];
  };

  const tableCells = (line) => {
    let trimmed = line.trim();
    if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
    if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
    return trimmed.split("|").map((cell) => cell.trim());
  };

  const isSeparatorRow = (line) => {
    const cells = tableCells(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, "")));
  };

  const flushTable = () => {
    if (!tableBuffer.length) return;
    const rows = tableBuffer.filter((line) => !isSeparatorRow(line)).map(tableCells);
    tableBuffer = [];
    if (!rows.length) return;

    const columnCount = Math.max(...rows.map((row) => row.length));
    const normalized = rows.map((row) => {
      const next = [...row];
      while (next.length < columnCount) next.push("");
      return next;
    });
    const [head, ...body] = normalized;

    html += '<div class="table-scroll"><table>';
    html += `<thead><tr>${head.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead>`;
    if (body.length) {
      html += `<tbody>${body
        .map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`)
        .join("")}</tbody>`;
    }
    html += "</table></div>";
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        closeList();
        inCode = true;
        codeBuffer = [];
      }
      continue;
    }

    if (inCode) {
      codeBuffer.push(line);
      continue;
    }

    const trimmed = line.trim();

    if (inMath) {
      if (trimmed === "$$") {
        flushMath();
        inMath = false;
      } else {
        mathBuffer.push(line);
      }
      continue;
    }

    if (trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4) {
      closeList();
      flushTable();
      const expression = trimmed.slice(2, -2).trim();
      html += `<div class="math-block">\\[${escapeHtml(expression)}\\]</div>`;
      continue;
    }

    if (trimmed === "$$") {
      closeList();
      flushTable();
      inMath = true;
      mathBuffer = [];
      continue;
    }

    if (!trimmed) {
      closeList();
      flushTable();
      continue;
    }

    if (trimmed.startsWith("|")) {
      closeList();
      tableBuffer.push(line);
      continue;
    }

    flushTable();

    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length, 3);
      html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
      continue;
    }

    const bullet = /^\s*[-*]\s+(.+)$/.exec(line);
    const number = /^\s*\d+\.\s+(.+)$/.exec(line);
    if (bullet || number) {
      const nextType = number ? "ol" : "ul";
      if (!inList || listType !== nextType) {
        closeList();
        listType = nextType;
        html += `<${listType}>`;
        inList = true;
      }
      html += `<li>${inlineMarkdown((bullet || number)[1])}</li>`;
      continue;
    }

    closeList();
    html += `<p>${inlineMarkdown(line)}</p>`;
  }

  if (inCode) flushCode();
  if (inMath) flushMath();
  flushTable();
  closeList();

  if (String(markdown || "").split("\n").length > 520) {
    html += "<p><strong>内容较长，已在网页中截断。</strong>请打开 GitHub 原文查看完整 SKILL.md。</p>";
  }

  return html;
}

function skillMatches(skill) {
  const query = state.query.trim().toLowerCase();
  const activeFilter = filterOptions.find((filter) => filter.id === state.filter) || filterOptions[0];
  const haystack = [
    skill.name,
    skill.displayName,
    skill.originalName,
    skill.description,
    skill.path,
    ...(skill.keywords || []),
  ]
    .join(" ")
    .toLowerCase();

  const queryMatch = !query || query.split(/\s+/).every((term) => haystack.includes(term));
  const filterMatch =
    activeFilter.type === "all"
      ? true
      : activeFilter.type === "source"
      ? skill.source === activeFilter.value
      : skill.category === activeFilter.value;
  return queryMatch && filterMatch;
}

function npxCommand(names) {
  if (!names.length) return "";
  if (names.length === 1) {
    return `npx skills add ${data.repo} -s ${names[0]}`;
  }
  return shellJoinCommand(`npx skills add ${data.repo} -s`, names);
}

function shellJoinCommand(command, args) {
  if (args.length <= 1) {
    return `${command} ${args.join(" ")}`.trim();
  }
  return `${command} \\\n${args.map((arg) => `    ${arg} \\`).join("\n")}`.replace(/ \\$/, "");
}

function bundleNpxCommand(bundle) {
  return `npx skills add ${bundle.githubUrl} -s '*'`;
}

function skillsShUrl(name = "") {
  const path = name ? `/${name}` : "";
  return `https://skills.sh/${data.repo}${path}`;
}

function skillsShLinks(names) {
  return names.map((name) => skillsShUrl(name)).join("\n");
}

function agentPrompt(skills) {
  const lines = skills.map((skill) => `- ${skill.displayName} (canonical: ${skill.name}, path: ${skill.path})`).join("\n");
  return `请帮我从 GitHub 仓库 ${data.repo} 安装以下 Awesome Ascend Skills：\n${lines}\n\n优先参考 skills.sh 页面确认 skill 信息：\n${skillsShLinks(skills.map((skill) => skill.name))}\n\n然后执行 npx 安装：\n${npxCommand(skills.map((skill) => skill.name))}\n\n如果 npx 不可用，请下载仓库后将对应目录复制到当前项目的 .agents/skills/。安装后请检查每个 SKILL.md 都存在，并简要确认安装位置。`;
}

function bundleAgentPrompt(bundle, skills) {
  const lines = skills.map((skill) => `- ${skill.displayName} (canonical: ${skill.name}, path: ${skill.path})`).join("\n");
  return `请帮我安装 Awesome Ascend Skills 的完整配套 Skill 组：${bundle.name}\n\nGitHub bundle 页面：\n${bundle.githubUrl}\n\n包含 Skills：\n${lines}\n\n请优先执行：\n${bundleNpxCommand(bundle)}\n\n安装后请检查对应 SKILL.md 都存在，并简要确认安装位置。`;
}

function selectedSkills() {
  const byName = new Map(data.skills.map((skill) => [skill.name, skill]));
  return [...state.selected].map((name) => byName.get(name)).filter(Boolean);
}

function filteredSkills() {
  return data.skills.filter(skillMatches);
}

function bundlesForSkill(skillName) {
  return data.bundles.filter((bundle) => bundle.skills.includes(skillName));
}

function selectBundle(bundleName) {
  const bundle = data.bundles.find((item) => item.name === bundleName);
  if (!bundle) return;
  bundle.skills.forEach((name) => state.selected.add(name));
  renderSkillList();
  renderBulkCommands();
  showToast(`已全选 ${bundle.name} 的 ${bundle.skills.length} 个 Skills`);
}

function selectAllVisibleSkills() {
  const skills = filteredSkills();
  skills.forEach((skill) => state.selected.add(skill.name));
  renderSkillList();
  renderBulkCommands();
  showToast(`已全选当前列表的 ${skills.length} 个 Skills`);
}

function clearVisibleSkills() {
  const skills = filteredSkills();
  skills.forEach((skill) => state.selected.delete(skill.name));
  renderSkillList();
  renderBulkCommands();
  showToast(`已取消当前列表的 ${skills.length} 个 Skills`);
}

function toggleSkillSelection(name) {
  if (state.selected.has(name)) {
    state.selected.delete(name);
    showToast("已取消选择该 Skill");
  } else {
    state.selected.add(name);
    showToast("已选择该 Skill，可到下方复制安装命令");
  }
  renderSkillList();
  renderBulkCommands();
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.remove("show"), 1800);
}

async function copyText(text, message = "已复制到剪贴板") {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
  showToast(message);
}

function renderFilters() {
  els.categoryFilters.innerHTML = filterOptions
    .map((filter) => `<button class="segment-button" data-filter="${filter.id}" aria-pressed="${filter.id === state.filter}">${filter.label}</button>`)
    .join("");
}

function renderSkillList() {
  const filtered = filteredSkills();
  if (!state.activeSkillName && filtered[0]) state.activeSkillName = filtered[0].name;
  if (!filtered.some((skill) => skill.name === state.activeSkillName) && filtered[0]) {
    state.activeSkillName = filtered[0].name;
  }

  els.resultCount.textContent = `${filtered.length} results`;
  els.selectedCount.textContent = `${state.selected.size} selected`;
  els.selectAllVisibleButton.disabled = filtered.length === 0;
  els.clearVisibleButton.disabled = filtered.length === 0;
  els.skillList.innerHTML = filtered
    .map((skill) => {
      const active = skill.name === state.activeSkillName ? " active" : "";
      const checked = state.selected.has(skill.name) ? "checked" : "";
      const alias = skill.originalName ? `<span class="tag">alias: ${escapeHtml(skill.originalName)}</span>` : "";
      return `<button class="skill-card${active}" data-skill="${escapeHtml(skill.name)}">
        <input type="checkbox" data-select-skill="${escapeHtml(skill.name)}" ${checked} aria-label="选择 ${escapeHtml(skill.displayName)}" />
        <span>
          <h3>${escapeHtml(skill.displayName)}</h3>
          <p>${escapeHtml(skill.description || "No description")}</p>
          <span class="tag-row">
            <span class="tag category-${escapeHtml(skill.category)}">${escapeHtml(categoryLabels[skill.category] || skill.category)}</span>
            <span class="tag">${escapeHtml(skill.source)}</span>
            ${alias}
          </span>
        </span>
      </button>`;
    })
    .join("");

  if (!filtered.length) {
    els.skillList.innerHTML = `<div class="empty-state">没有匹配的 Skill。换一个关键词或清空筛选。</div>`;
    els.skillDetail.innerHTML = `<div class="detail-header"><h2>没有结果</h2><p>当前筛选条件没有匹配项。</p></div>`;
  } else {
    renderSkillDetail(data.skills.find((skill) => skill.name === state.activeSkillName) || filtered[0]);
  }
}

function renderSkillDetail(skill) {
  if (!skill) return;
  const singleNpx = npxCommand([skill.name]);
  const singleSkillsSh = skillsShUrl(skill.name);
  const singlePrompt = agentPrompt([skill]);
  const relatedBundles = bundlesForSkill(skill.name);
  const isSelected = state.selected.has(skill.name);
  const externalNotice = skill.syncedFrom
    ? `<div class="external-notice">
        <strong>该 SKILL 为外部 skill</strong>
        <span>原链接：<a href="${escapeHtml(skill.syncedFrom)}">${escapeHtml(skill.syncedFrom)}</a></span>
      </div>`
    : "";
  const bundleBlock = relatedBundles.length
    ? `<div class="bundle-callout">
        <strong>所属配套 Skill 组</strong>
        ${relatedBundles
          .map(
            (bundle) => `<div class="bundle-callout-row">
              <span>${escapeHtml(bundle.name)} · ${bundle.skills.length} skills</span>
              <span class="bundle-callout-actions">
                <a class="small-copy" href="${escapeHtml(bundle.githubUrl)}">打开 GitHub</a>
                <button class="small-copy" data-select-bundle="${escapeHtml(bundle.name)}">全选该组</button>
              </span>
            </div>`
          )
          .join("")}
      </div>`
    : "";
  els.skillDetail.innerHTML = `
    <div class="detail-header">
      <div class="tag-row">
        <span class="tag category-${escapeHtml(skill.category)}">${escapeHtml(categoryLabels[skill.category] || skill.category)}</span>
        <span class="tag">${escapeHtml(skill.source)}</span>
        ${skill.originalName ? `<span class="tag">canonical: ${escapeHtml(skill.name)}</span>` : ""}
      </div>
      <h2>${escapeHtml(skill.displayName)}</h2>
      ${externalNotice}
      <p>${escapeHtml(skill.description || "No description")}</p>
      <div class="detail-actions">
        <button class="button ${isSelected ? "secondary" : "primary"}" data-toggle-skill="${escapeHtml(skill.name)}">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="${isSelected ? "M5 12h14" : "M5 12h14M12 5v14"}" /></svg>
          ${isSelected ? "取消选择" : "选择此 Skill"}
        </button>
        <a class="button secondary" href="#install">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12l7 7 7-7" /></svg>
          查看安装命令
        </a>
        <span class="detail-action-break" aria-hidden="true"></span>
        <button class="button secondary" data-copy="${escapeHtml(singleNpx)}" data-copy-message="已复制下载链接">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 7h8M8 12h8M8 17h5" /></svg>
          复制下载链接
        </button>
        <a class="button secondary" href="${escapeHtml(skill.githubUrl)}">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M8 7h9v9" /></svg>
          GitHub
        </a>
        <a class="button secondary" href="${escapeHtml(singleSkillsSh)}">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M8 7h9v9" /></svg>
          打开 skills.sh
        </a>
        <button class="button secondary" data-copy="${escapeHtml(singleSkillsSh)}" data-copy-message="已复制 skills.sh 页面链接">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8h10v10H8zM6 16H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" /></svg>
          复制链接
        </button>
        <button class="button secondary" data-copy="${escapeHtml(singlePrompt)}" data-copy-message="已复制 Agent 下载说明">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 8h10M7 12h6M5 20l3-3h11V5H5v15Z" /></svg>
          复制 Agent 下载说明
        </button>
      </div>
      ${bundleBlock}
    </div>
    <div class="detail-body">
      <div class="markdown-body">${renderMarkdown(skill.body)}</div>
    </div>
  `;
  typesetMath(els.skillDetail);
}

function typesetMath(root) {
  if (!window.MathJax || !window.MathJax.typesetPromise) return;
  window.MathJax.typesetPromise([root]).catch((error) => {
    console.warn("MathJax typeset failed", error);
  });
}

function renderBulkCommands() {
  const skills = selectedSkills();
  if (!skills.length) {
    els.bulkNpx.innerHTML = "<code>先选择一个或多个 Skill</code>";
    els.bulkSkillsSh.innerHTML = "<code>先选择一个或多个 Skill</code>";
    els.bulkAgentPrompt.innerHTML = "<code>先选择一个或多个 Skill</code>";
    return;
  }

  const names = skills.map((skill) => skill.name);
  els.bulkNpx.innerHTML = `<code>${escapeHtml(npxCommand(names))}</code>`;
  els.bulkSkillsSh.innerHTML = `<code>${escapeHtml(skillsShLinks(names))}</code>`;
  els.bulkAgentPrompt.innerHTML = `<code>${escapeHtml(agentPrompt(skills))}</code>`;
}

function renderBundles() {
  els.bundleGrid.innerHTML = data.bundles
    .map((bundle) => {
      const names = bundle.skills;
      const promptSkills = names
        .map((name) => data.skills.find((skill) => skill.name === name))
        .filter(Boolean);
      return `<article class="bundle-card">
        <div class="tag-row">
          <span class="tag category-${escapeHtml(bundle.category)}">${escapeHtml(categoryLabels[bundle.category] || bundle.category)}</span>
          <span class="tag">${bundle.skills.length} skills</span>
        </div>
        <h3>${escapeHtml(bundle.name)}</h3>
        <p>${escapeHtml(bundle.description)}</p>
        <div class="bundle-skills">
          ${bundle.displaySkills.map((name) => `<span class="tag">${escapeHtml(name)}</span>`).join("")}
        </div>
        <div class="card-actions">
          <button class="button primary" data-copy="${escapeHtml(bundleNpxCommand(bundle))}" data-copy-message="已复制下载链接">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 7h8M8 12h8M8 17h5" /></svg>
            复制下载链接
          </button>
          <a class="button secondary" href="${escapeHtml(bundle.githubUrl)}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M8 7h9v9" /></svg>
            打开 GitHub
          </a>
          <button class="button secondary" data-copy="${escapeHtml(bundleAgentPrompt(bundle, promptSkills))}" data-copy-message="已复制 Agent 安装说明">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 8h10M7 12h6M5 20l3-3h11V5H5v15Z" /></svg>
            复制 Agent 安装说明
          </button>
        </div>
      </article>`;
    })
    .join("");
}

function render() {
  els.statsLine.textContent = `${data.stats.skillCount || data.skills.length} 个 Skills，${data.stats.bundleCount || data.bundles.length} 个配套组。支持搜索、详情预览、npx / skills.sh / Agent 安装说明复制。`;
  renderFilters();
  renderSkillList();
  renderBulkCommands();
  renderBundles();
}

document.addEventListener("input", (event) => {
  if (event.target === els.searchInput) {
    state.query = els.searchInput.value;
    renderSkillList();
  }
});

document.addEventListener("click", (event) => {
  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) {
    copyText(copyButton.getAttribute("data-copy"), copyButton.getAttribute("data-copy-message") || "复制成功");
    return;
  }

  const copyTarget = event.target.closest("[data-copy-target]");
  if (copyTarget) {
    const target = document.querySelector(`#${copyTarget.getAttribute("data-copy-target")}`);
    copyText(target?.innerText || "", copyTarget.getAttribute("data-copy-message") || "复制成功");
    return;
  }

  const filterButton = event.target.closest("[data-filter]");
  if (filterButton) {
    state.filter = filterButton.getAttribute("data-filter");
    render();
    return;
  }

  const bundleSelectButton = event.target.closest("[data-select-bundle]");
  if (bundleSelectButton) {
    selectBundle(bundleSelectButton.getAttribute("data-select-bundle"));
    return;
  }

  const toggleButton = event.target.closest("[data-toggle-skill]");
  if (toggleButton) {
    toggleSkillSelection(toggleButton.getAttribute("data-toggle-skill"));
    return;
  }

  const checkbox = event.target.closest("[data-select-skill]");
  if (checkbox) {
    const name = checkbox.getAttribute("data-select-skill");
    if (checkbox.checked) state.selected.add(name);
    else state.selected.delete(name);
    renderSkillList();
    renderBulkCommands();
    event.stopPropagation();
    return;
  }

  const card = event.target.closest("[data-skill]");
  if (card) {
    state.activeSkillName = card.getAttribute("data-skill");
    renderSkillList();
    return;
  }

  if (event.target.closest("#clearSelectionButton")) {
    state.selected.clear();
    renderSkillList();
    renderBulkCommands();
  }

  if (event.target.closest("#selectAllVisibleButton")) {
    selectAllVisibleSkills();
  }

  if (event.target.closest("#clearVisibleButton")) {
    clearVisibleSkills();
  }
});

render();

if (!window.location.hash) {
  window.scrollTo(0, 0);
  window.addEventListener("load", () => window.scrollTo(0, 0), { once: true });
  window.addEventListener("pageshow", () => window.scrollTo(0, 0), { once: true });
}
