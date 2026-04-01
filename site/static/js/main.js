// ── Geometric background ────────────────────────────
(function () {
  const canvas = document.getElementById('geo-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, nodes = [], mouse = { x: -1000, y: -1000 };

  function resize() { w = canvas.width = window.innerWidth; h = canvas.height = window.innerHeight; }

  function initNodes() {
    const count = Math.min(Math.floor((w * h) / 30000), 60);
    nodes = [];
    for (let i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
        r: Math.random() * 1.2 + 0.4
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(184, 134, 11, 0.2)';
      ctx.fill();
    }
    const maxDist = 140;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(184, 134, 11, ${(1 - dist / maxDist) * 0.06})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
      const mdx = nodes[i].x - mouse.x, mdy = nodes[i].y - mouse.y;
      const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
      if (mdist < 180) {
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(mouse.x, mouse.y);
        ctx.strokeStyle = `rgba(184, 134, 11, ${(1 - mdist / 180) * 0.12})`;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', () => { resize(); initNodes(); });
  window.addEventListener('mousemove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  resize(); initNodes(); draw();
})();

// ── Scroll reveal ───────────────────────────────────
(function () {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add('visible'); });
  }, { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
})();

// ── Bar animation ───────────────────────────────────
(function () {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting || e.target.dataset.animated) return;
      e.target.dataset.animated = '1';
      setTimeout(() => { e.target.style.width = e.target.dataset.width + '%'; }, 80);
    });
  }, { threshold: 0.3 });
  document.querySelectorAll('[data-width]').forEach((f) => observer.observe(f));
})();

// ── Freivalds demo ──────────────────────────────────
const P = 4294967291;
function mulmod(a, b, m) { return Number((BigInt(a) * BigInt(b)) % BigInt(m)); }
function addmod(a, b, m) { return (a + b) % m; }
function matVecMod(M, v, m) {
  return M.map(row => row.reduce((s, val, j) => addmod(s, mulmod(val, v[j], m), m), 0));
}
function dotMod(a, b, m) { return a.reduce((s, v, i) => addmod(s, mulmod(v, b[i], m), m), 0); }
function randVec(n) { return Array.from({ length: n }, () => Math.floor(Math.random() * 1000)); }
function randMat(n) { return Array.from({ length: n }, () => randVec(n)); }
function transpose(M) { return M[0].map((_, i) => M.map(r => r[i])); }

window.runFreivalds = function (honest) {
  const n = 3;
  const W = randMat(n), x = randVec(n);
  const z_real = matVecMod(W, x, P);
  const z = z_real.slice();
  if (!honest) z[Math.floor(Math.random() * n)] = Math.floor(Math.random() * 1000);

  const r = randVec(n);
  const v = matVecMod(transpose(W), r, P);
  const lhs = dotMod(v, x, P), rhs = dotMod(r, z, P);
  const pass = lhs === rhs;

  document.getElementById('demo-eq').innerHTML =
    `<span class="dim">r</span> = [${r.join(', ')}]\n` +
    `<span class="dim">x</span> = [${x.join(', ')}]\n` +
    `<span class="dim">z</span> = [${z.join(', ')}]` +
    (honest ? '' : ' <span class="err">(tampered)</span>') + '\n\n' +
    `<span class="val">${lhs}</span> <span class="dim">=?</span> <span class="val">${rhs}</span>`;

  const res = document.getElementById('demo-result');
  res.style.display = 'block';
  if (pass) {
    res.className = 'demo-result pass';
    res.textContent = honest ? 'Check passed — honest computation verified' : 'False accept (prob \u2264 1/p)';
  } else {
    res.className = 'demo-result fail';
    res.textContent = 'Check failed — tampering detected';
  }
};

// ── Smooth scroll ───────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach((a) => {
  a.addEventListener('click', (e) => {
    const t = document.querySelector(a.getAttribute('href'));
    if (t) { e.preventDefault(); t.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
});
