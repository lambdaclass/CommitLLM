// ── Bar widths (set immediately, no animation) ─────
document.querySelectorAll('[data-width]').forEach(function (el) {
  el.style.width = el.dataset.width + '%';
});

// ── Freivalds demo ──────────────────────────────────
var P = 4294967291;
function mulmod(a, b, m) { return Number((BigInt(a) * BigInt(b)) % BigInt(m)); }
function addmod(a, b, m) { return (a + b) % m; }
function matVecMod(M, v, m) {
  return M.map(function (row) {
    return row.reduce(function (s, val, j) { return addmod(s, mulmod(val, v[j], m), m); }, 0);
  });
}
function dotMod(a, b, m) {
  return a.reduce(function (s, v, i) { return addmod(s, mulmod(v, b[i], m), m); }, 0);
}
function randVec(n) { return Array.from({ length: n }, function () { return Math.floor(Math.random() * 1000); }); }
function randMat(n) { return Array.from({ length: n }, function () { return randVec(n); }); }
function transpose(M) { return M[0].map(function (_, i) { return M.map(function (r) { return r[i]; }); }); }

window.runFreivalds = function (honest) {
  var n = 3;
  var W = randMat(n), x = randVec(n);
  var z_real = matVecMod(W, x, P);
  var z = z_real.slice();
  if (!honest) z[Math.floor(Math.random() * n)] = Math.floor(Math.random() * 1000);

  var r = randVec(n);
  var v = matVecMod(transpose(W), r, P);
  var lhs = dotMod(v, x, P), rhs = dotMod(r, z, P);
  var pass = lhs === rhs;

  document.getElementById('demo-eq').innerHTML =
    '<span class="dim">r</span> = [' + r.join(', ') + ']\n' +
    '<span class="dim">x</span> = [' + x.join(', ') + ']\n' +
    '<span class="dim">z</span> = [' + z.join(', ') + ']' +
    (honest ? '' : ' <span class="err">(tampered)</span>') + '\n\n' +
    '<span class="val">' + lhs + '</span> <span class="dim">=?</span> <span class="val">' + rhs + '</span>';

  var res = document.getElementById('demo-result');
  res.style.display = 'block';
  if (pass) {
    res.className = 'demo-result pass';
    res.textContent = honest ? 'Check passed \u2014 honest computation verified' : 'False accept (prob \u2264 1/p)';
  } else {
    res.className = 'demo-result fail';
    res.textContent = 'Check failed \u2014 tampering detected';
  }
};

// ── Smooth scroll ───────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(function (a) {
  a.addEventListener('click', function (e) {
    var t = document.querySelector(a.getAttribute('href'));
    if (t) { e.preventDefault(); t.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
});
